//! A session's RTP and RTCP sockets, bound together and held until the
//! session takes them.
//!
//! A SIP endpoint has to name its RTP port in the SDP before the media
//! session starts: in the offer it sends, or in the answer it builds while
//! the call rings. Choosing that port by binding it, closing the socket, and
//! binding it again when the session starts leaves a window in which any other
//! bind in the process (or on the machine) can be given the same port, and
//! the session then fails with "address in use". Windows makes this likely,
//! because it hands a just-released ephemeral port straight back to the next
//! bind (os error 10048). [`RtpSockets`] closes that window: the port is
//! chosen by binding it, stays bound, and [`crate::MediaSession`] starts on the
//! sockets themselves.

use std::fmt;
use tokio::net::UdpSocket;

use crate::error::{Error, Result};

/// How many ephemeral ports [`RtpSockets::allocate`] tries for an even one
/// with its RTCP neighbour free before giving up. Each miss costs one bind.
const ALLOCATE_ATTEMPTS: usize = 32;

/// An RTP socket and its RTCP socket on the port above it (RFC 3550 §11),
/// both bound on all interfaces.
///
/// Dropping it releases both ports. Passing it to one of the
/// `MediaSession::start_on_sockets*` functions hands the sockets to the
/// session, with no moment in between when either port is free.
pub struct RtpSockets {
    rtp: UdpSocket,
    rtcp: UdpSocket,
    port: u16,
}

impl RtpSockets {
    /// Bind an even ephemeral port chosen by the OS, and the odd port above
    /// it, keeping both bound.
    ///
    /// An odd port, or an even one whose neighbour is taken, is released and
    /// another tried, up to a fixed number of attempts.
    pub async fn allocate() -> Result<Self> {
        let mut last_error = None;
        for _ in 0..ALLOCATE_ATTEMPTS {
            let rtp = UdpSocket::bind("0.0.0.0:0").await?;
            let port = rtp.local_addr()?.port();
            if port % 2 != 0 {
                last_error = Some(format!("port {port} is odd"));
                continue;
            }
            match UdpSocket::bind(("0.0.0.0", port + 1)).await {
                Ok(rtcp) => return Ok(Self { rtp, rtcp, port }),
                Err(e) => last_error = Some(format!("RTCP port {} is taken: {e}", port + 1)),
            }
        }
        Err(Error::Network(std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            format!(
                "no even RTP port with a free RTCP port in {ALLOCATE_ATTEMPTS} attempts (last: {})",
                last_error.unwrap_or_default()
            ),
        )))
    }

    /// Bind `rtp_port` for RTP and `rtp_port + 1` for RTCP, on all interfaces.
    pub async fn bind(rtp_port: u16) -> Result<Self> {
        let rtcp_port = rtp_port.checked_add(1).ok_or_else(|| {
            Error::invalid_parameter(format!("RTP port {rtp_port} has no port above it for RTCP"))
        })?;
        let rtp = UdpSocket::bind(("0.0.0.0", rtp_port)).await?;
        let port = rtp.local_addr()?.port();
        let rtcp = UdpSocket::bind(("0.0.0.0", rtcp_port)).await?;
        Ok(Self { rtp, rtcp, port })
    }

    /// The RTP port; RTCP is on the one above it.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The RTP socket, for traffic that must come from the media port before
    /// the session starts, such as a STUN binding request.
    pub fn rtp(&self) -> &UdpSocket {
        &self.rtp
    }

    pub(crate) fn into_parts(self) -> (UdpSocket, UdpSocket) {
        (self.rtp, self.rtcp)
    }
}

impl fmt::Debug for RtpSockets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RtpSockets")
            .field("rtp_port", &self.port)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn allocate_holds_an_even_port_and_the_one_above_it() {
        let sockets = RtpSockets::allocate().await.expect("a port pair");
        let port = sockets.port();
        assert_eq!(port % 2, 0, "RTP port {port} is odd");
        assert_eq!(sockets.rtp().local_addr().unwrap().port(), port);
        assert!(
            std::net::UdpSocket::bind(("0.0.0.0", port)).is_err(),
            "RTP port {port} was free while held"
        );
        assert!(
            std::net::UdpSocket::bind(("0.0.0.0", port + 1)).is_err(),
            "RTCP port {} was free while held",
            port + 1
        );
    }

    #[tokio::test]
    async fn dropping_the_sockets_releases_both_ports() {
        let sockets = RtpSockets::allocate().await.expect("a port pair");
        let port = sockets.port();
        drop(sockets);
        let again = RtpSockets::bind(port).await.expect("both ports free again");
        assert_eq!(again.port(), port);
    }

    #[tokio::test]
    async fn bind_fails_when_the_rtcp_port_is_taken() {
        let sockets = RtpSockets::allocate().await.expect("a port pair");
        let port = sockets.port();
        let (rtp, rtcp) = sockets.into_parts();
        drop(rtp);
        assert!(RtpSockets::bind(port).await.is_err());
        drop(rtcp);
    }

    /// The session runs on the very sockets it was given: the ports never
    /// come free between allocation and start.
    #[tokio::test]
    async fn a_session_started_on_the_sockets_keeps_their_ports() {
        let sockets = RtpSockets::allocate().await.expect("a port pair");
        let port = sockets.port();
        let session = crate::MediaSession::start_on_sockets(
            sockets,
            "127.0.0.1:9".parse().unwrap(),
            crate::CodecType::Pcmu,
        )
        .expect("a session on the held ports");
        assert!(std::net::UdpSocket::bind(("0.0.0.0", port)).is_err());
        assert!(std::net::UdpSocket::bind(("0.0.0.0", port + 1)).is_err());
        session.stop();
    }

    #[tokio::test]
    async fn bind_rejects_the_last_port() {
        assert!(matches!(
            RtpSockets::bind(u16::MAX).await,
            Err(Error::InvalidParameter(_))
        ));
    }
}

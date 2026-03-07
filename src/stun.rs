//! STUN client for NAT traversal.
//!
//! Implements RFC 5389 STUN (Session Traversal Utilities for NAT) to discover
//! the public IP address and port as seen by a STUN server.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use tokio::net::UdpSocket;
use tokio::time::{Duration, timeout};

use crate::error::{Error, Result};

/// Default STUN servers (Google's public STUN servers).
pub const DEFAULT_STUN_SERVERS: &[&str] = &[
    "stun.l.google.com:19302",
    "stun1.l.google.com:19302",
    "stun2.l.google.com:19302",
    "stun3.l.google.com:19302",
    "stun4.l.google.com:19302",
];

/// STUN message types
const STUN_BINDING_REQUEST: u16 = 0x0001;
const STUN_BINDING_RESPONSE: u16 = 0x0101;

/// STUN magic cookie (RFC 5389)
const STUN_MAGIC_COOKIE: u32 = 0x2112_A442;

/// STUN attribute types
const ATTR_MAPPED_ADDRESS: u16 = 0x0001;
const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;

/// Result of a STUN binding request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StunResult {
    /// The public IP address as seen by the STUN server.
    pub public_ip: IpAddr,
    /// The public port as seen by the STUN server.
    pub public_port: u16,
    /// The local IP address used.
    pub local_ip: IpAddr,
    /// The local port used.
    pub local_port: u16,
}

impl StunResult {
    /// Get the public socket address.
    pub fn public_addr(&self) -> SocketAddr {
        SocketAddr::new(self.public_ip, self.public_port)
    }

    /// Get the local socket address.
    pub fn local_addr(&self) -> SocketAddr {
        SocketAddr::new(self.local_ip, self.local_port)
    }

    /// Check if we're behind NAT (public != local).
    pub fn is_natted(&self) -> bool {
        self.public_ip != self.local_ip || self.public_port != self.local_port
    }
}

/// Discover public IP/port using STUN with a specific socket.
///
/// This allows you to use an already-bound socket (e.g., your RTP socket)
/// to discover what public address that socket appears as to the outside world.
///
/// # Arguments
/// * `socket` - The UDP socket to use for STUN discovery
/// * `stun_server` - The STUN server address (e.g., "stun.l.google.com:19302")
/// * `timeout_ms` - Timeout in milliseconds
pub async fn discover_with_socket(
    socket: &UdpSocket,
    stun_server: &str,
    timeout_ms: u64,
) -> Result<StunResult> {
    // Resolve STUN server
    let stun_addr: SocketAddr = tokio::net::lookup_host(stun_server)
        .await
        .map_err(|e| Error::Network(e))?
        .next()
        .ok_or_else(|| Error::stun(format!("Could not resolve STUN server: {}", stun_server)))?;

    // Build STUN binding request
    let transaction_id: [u8; 12] = rand::random();
    let request = build_binding_request(&transaction_id);

    // Send request
    socket
        .send_to(&request, stun_addr)
        .await
        .map_err(|e| Error::Network(e))?;

    // Receive response with timeout
    let mut buf = [0u8; 512];
    let (len, _from) = timeout(
        Duration::from_millis(timeout_ms),
        socket.recv_from(&mut buf),
    )
    .await
    .map_err(|_| Error::stun("STUN request timed out"))?
    .map_err(|e| Error::Network(e))?;

    // Parse response
    let (public_ip, public_port) = parse_binding_response(&buf[..len], &transaction_id)?;

    let local_addr = socket.local_addr().map_err(|e| Error::Network(e))?;

    Ok(StunResult {
        public_ip,
        public_port,
        local_ip: local_addr.ip(),
        local_port: local_addr.port(),
    })
}

/// Discover public IP/port using STUN with default servers.
///
/// Creates a temporary socket, queries the STUN server, and returns the
/// discovered public address. Tries multiple servers if the first fails.
///
/// # Arguments
/// * `local_port` - Local port to bind (0 for OS-assigned)
pub async fn discover(local_port: u16) -> Result<StunResult> {
    let socket = UdpSocket::bind(format!("0.0.0.0:{}", local_port))
        .await
        .map_err(|e| Error::Network(e))?;

    discover_public_address(&socket).await
}

/// Discover public address using the given socket, trying multiple STUN servers.
pub async fn discover_public_address(socket: &UdpSocket) -> Result<StunResult> {
    let mut last_error = None;

    for server in DEFAULT_STUN_SERVERS {
        match discover_with_socket(socket, server, 3000).await {
            Ok(result) => {
                log::info!(
                    "STUN discovery via {}: local {}:{} -> public {}:{}",
                    server,
                    result.local_ip,
                    result.local_port,
                    result.public_ip,
                    result.public_port
                );
                return Ok(result);
            }
            Err(e) => {
                log::warn!("STUN server {} failed: {}", server, e);
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| Error::stun("All STUN servers failed")))
}

/// Build a STUN binding request message.
fn build_binding_request(transaction_id: &[u8; 12]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(20);

    // Message type: Binding Request (0x0001)
    msg.extend_from_slice(&STUN_BINDING_REQUEST.to_be_bytes());

    // Message length: 0 (no attributes in request)
    msg.extend_from_slice(&0u16.to_be_bytes());

    // Magic cookie
    msg.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());

    // Transaction ID (12 bytes)
    msg.extend_from_slice(transaction_id);

    msg
}

/// Parse a STUN binding response to extract the mapped address.
fn parse_binding_response(data: &[u8], expected_txn_id: &[u8; 12]) -> Result<(IpAddr, u16)> {
    if data.len() < 20 {
        return Err(Error::stun("Response too short"));
    }

    // Check message type
    let msg_type = u16::from_be_bytes([data[0], data[1]]);
    if msg_type != STUN_BINDING_RESPONSE {
        return Err(Error::stun(format!(
            "Unexpected message type: 0x{:04x}",
            msg_type
        )));
    }

    // Check magic cookie
    let cookie = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    if cookie != STUN_MAGIC_COOKIE {
        return Err(Error::stun("Invalid magic cookie"));
    }

    // Check transaction ID
    if &data[8..20] != expected_txn_id {
        return Err(Error::stun("Transaction ID mismatch"));
    }

    // Parse message length
    let msg_len = u16::from_be_bytes([data[2], data[3]]) as usize;
    if data.len() < 20 + msg_len {
        return Err(Error::stun("Truncated message"));
    }

    // Parse attributes
    let mut offset = 20;
    while offset + 4 <= 20 + msg_len {
        let attr_type = u16::from_be_bytes([data[offset], data[offset + 1]]);
        let attr_len = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
        offset += 4;

        if offset + attr_len > data.len() {
            break;
        }

        match attr_type {
            ATTR_XOR_MAPPED_ADDRESS => {
                return parse_xor_mapped_address(&data[offset..offset + attr_len]);
            }
            ATTR_MAPPED_ADDRESS => {
                return parse_mapped_address(&data[offset..offset + attr_len]);
            }
            _ => {}
        }

        // Pad to 4-byte boundary
        offset += (attr_len + 3) & !3;
    }

    Err(Error::stun("No mapped address in response"))
}

/// Parse XOR-MAPPED-ADDRESS attribute.
fn parse_xor_mapped_address(data: &[u8]) -> Result<(IpAddr, u16)> {
    if data.len() < 8 {
        return Err(Error::stun("XOR-MAPPED-ADDRESS too short"));
    }

    let family = data[1];
    let xport = u16::from_be_bytes([data[2], data[3]]);
    let port = xport ^ ((STUN_MAGIC_COOKIE >> 16) as u16);

    match family {
        0x01 => {
            // IPv4
            let xaddr = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
            let addr = xaddr ^ STUN_MAGIC_COOKIE;
            let ip = Ipv4Addr::from(addr);
            Ok((IpAddr::V4(ip), port))
        }
        0x02 => {
            // IPv6
            if data.len() < 20 {
                return Err(Error::stun("XOR-MAPPED-ADDRESS IPv6 too short"));
            }
            // For IPv6, XOR with magic cookie + transaction ID
            // Simplified: just return error for now, IPv4 is most common
            Err(Error::stun("IPv6 STUN not yet supported"))
        }
        _ => Err(Error::stun(format!("Unknown address family: {}", family))),
    }
}

/// Parse MAPPED-ADDRESS attribute (legacy, non-XOR).
fn parse_mapped_address(data: &[u8]) -> Result<(IpAddr, u16)> {
    if data.len() < 8 {
        return Err(Error::stun("MAPPED-ADDRESS too short"));
    }

    let family = data[1];
    let port = u16::from_be_bytes([data[2], data[3]]);

    match family {
        0x01 => {
            // IPv4
            let ip = Ipv4Addr::new(data[4], data[5], data[6], data[7]);
            Ok((IpAddr::V4(ip), port))
        }
        0x02 => {
            // IPv6
            Err(Error::stun("IPv6 STUN not yet supported"))
        }
        _ => Err(Error::stun(format!("Unknown address family: {}", family))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_binding_request() {
        let txn_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let request = build_binding_request(&txn_id);

        assert_eq!(request.len(), 20);
        assert_eq!(&request[0..2], &[0x00, 0x01]); // Binding Request
        assert_eq!(&request[2..4], &[0x00, 0x00]); // Length 0
        assert_eq!(&request[4..8], &[0x21, 0x12, 0xA4, 0x42]); // Magic cookie
        assert_eq!(&request[8..20], &txn_id);
    }

    #[test]
    fn test_parse_xor_mapped_address() {
        // XOR-MAPPED-ADDRESS for 192.0.2.1:32853
        // Magic cookie: 0x2112A442
        // Port: 32853 = 0x8055, XOR'd with 0x2112 = 0xA147
        // IP: 192.0.2.1 = 0xC0000201, XOR'd with 0x2112A442 = 0xE112A643
        let data = [
            0x00, 0x01, // Reserved + Family (IPv4)
            0xA1, 0x47, // X-Port (32853 XOR 0x2112)
            0xE1, 0x12, 0xA6, 0x43, // X-Address (0xC0000201 XOR 0x2112A442)
        ];

        let (ip, port) = parse_xor_mapped_address(&data).unwrap();
        assert_eq!(ip, IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)));
        assert_eq!(port, 32853);
    }

    #[test]
    fn test_stun_result_is_natted() {
        let result = StunResult {
            public_ip: IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1)),
            public_port: 12345,
            local_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            local_port: 5000,
        };
        assert!(result.is_natted());

        let result_no_nat = StunResult {
            public_ip: IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1)),
            public_port: 5000,
            local_ip: IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1)),
            local_port: 5000,
        };
        assert!(!result_no_nat.is_natted());
    }
}

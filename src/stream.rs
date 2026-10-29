//! A headless media stream for one call: PCM frames in and out over a socket
//! the caller owns, with no audio device and no threads.
//!
//! [`MediaSession`](crate::MediaSession) is built for a softphone: it opens
//! the microphone and speaker and runs a thread for each. A server answering
//! many calls has neither, and wants every call on its own async task. This
//! is that shape:
//!
//! - [`MediaStream::recv_frame`] waits for the next event: 20 ms of decoded
//!   PCM at the codec's own rate (every 20 ms once audio has started, as
//!   received, concealed or silence), a DTMF digit, or a report that no RTP
//!   has arrived for a while.
//! - [`MediaStream::send_frame`] encodes and sends 20 ms of PCM, or the next
//!   packet of a queued DTMF digit in its place.
//! - [`MediaStream::rtcp_tick`] sends a sender report; [`MediaStream::stats`]
//!   reports counters.
//!
//! All the packet handling lives in [`StreamCore`], which does no I/O and
//! takes the time as a parameter, so it can be tested and fuzzed exactly.
//!
//! # Trust
//!
//! The stream sends to the address the SDP named until it learns where the
//! media really comes from (symmetric RTP), but it only learns that from a
//! packet it can trust: under SRTP, any packet that authenticated; without
//! SRTP, a packet from the host the SDP named. After that, packets from
//! anywhere else are dropped. SRTP packets that fail authentication or are
//! replays are dropped and counted, and change nothing.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use tokio::net::UdpSocket;

use crate::codec::{AudioDecoder, AudioEncoder, CodecType, create_decoder, create_encoder};
use crate::error::{Error, Result};
use crate::jitter::{JitterBuffer, JitterConfig, JitterMode, JitterStats};
use crate::rtp::{RtpHeader, build_rtcp_sr};
use crate::srtp::SrtpContext;

/// The SRTP contexts for one call: one to decrypt what arrives (keyed from the
/// peer's `a=crypto`), one to encrypt what is sent (keyed from ours).
pub struct SrtpPair {
    /// Unprotects incoming packets.
    pub inbound: SrtpContext,
    /// Protects outgoing packets.
    pub outbound: SrtpContext,
}

impl std::fmt::Debug for SrtpPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SrtpPair").finish_non_exhaustive()
    }
}

/// How a stream is set up, from the negotiated SDP.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// The audio codec.
    pub codec: CodecType,
    /// The payload type the audio is carried on, in both directions. Defaults
    /// to the codec's conventional number; set it for a dynamic payload type.
    pub payload_type: u8,
    /// The telephone-event payload type (RFC 4733), if negotiated.
    pub dtmf_payload_type: Option<u8>,
    /// The telephone-event clock rate. 8000 almost everywhere.
    pub dtmf_clock_rate: u32,
    /// Where the SDP said to send media.
    pub remote: SocketAddr,
    /// Jitter buffer behaviour. Defaults to adaptive at 40 ms, between 20 and
    /// 120: the far end of a server's call is usually a PBX one hop away, and
    /// every millisecond here is added to the response time.
    pub jitter: JitterMode,
    /// The most packets the jitter buffer holds.
    pub max_packets: usize,
    /// How long without any RTP before [`StreamEvent::Timeout`].
    pub rtp_timeout: Duration,
    /// How many 20 ms packets one DTMF digit lasts.
    pub dtmf_packets: u16,
}

impl StreamConfig {
    /// Defaults for `codec` sending to `remote`: the codec's own payload type,
    /// telephone-event on 101 at 8000 Hz, a 40/20/120 ms adaptive jitter
    /// buffer of at most 50 packets, a 5 s RTP timeout and 100 ms digits.
    #[must_use]
    pub fn new(codec: CodecType, remote: SocketAddr) -> Self {
        Self {
            codec,
            payload_type: codec.payload_type(),
            dtmf_payload_type: Some(101),
            dtmf_clock_rate: 8000,
            remote,
            jitter: JitterMode::Adaptive {
                target_ms: 40,
                min_ms: 20,
                max_ms: 120,
            },
            max_packets: 50,
            rtp_timeout: Duration::from_secs(5),
            dtmf_packets: 5,
        }
    }
}

/// What a 20 ms frame of received audio is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// Decoded from a packet that arrived.
    Received,
    /// Rebuilt for a lost packet: by the codec (Opus FEC or extrapolation) or
    /// as silence.
    Concealed,
    /// Nothing to play: the sender paused (silence suppression) or the
    /// network stalled. Zeros, so the frame clock keeps running.
    Silence,
}

/// 20 ms of received audio at the codec's PCM rate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    /// Mono samples, [`CodecType::pcm_samples_per_packet`] of them.
    pub pcm: Vec<i16>,
    /// Where they came from.
    pub kind: FrameKind,
}

/// A DTMF digit received as an RFC 4733 telephone-event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtmfEvent {
    /// `0`-`9`, `*`, `#` or `A`-`D`.
    pub digit: char,
    /// The RFC 4733 event code (0-15).
    pub event: u8,
}

/// What [`MediaStream::recv_frame`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// The next 20 ms of audio.
    Frame(AudioFrame),
    /// A digit, reported once per press, as its first packet arrives.
    Dtmf(DtmfEvent),
    /// No RTP has arrived for this long. Repeats every
    /// [`StreamConfig::rtp_timeout`] while the silence lasts.
    Timeout {
        /// Time since the last packet, or since the stream started.
        silent_for: Duration,
    },
}

/// Why an incoming datagram was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dropped {
    /// Not a well-formed RTP packet.
    Malformed,
    /// SRTP authentication failed: wrong key, tampered, or forged.
    Unauthenticated,
    /// SRTP replay: already received, or older than the replay window.
    Replayed,
    /// From an address this stream does not trust (see the module docs).
    UnexpectedSource,
    /// A payload type that is neither the audio nor telephone-event (comfort
    /// noise, say), or RTCP multiplexed on the RTP port.
    OtherPayload,
}

/// Counters for one stream.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// RTP packets accepted.
    pub packets_received: u64,
    /// RTP packets sent, DTMF included.
    pub packets_sent: u64,
    /// Bytes of accepted datagrams.
    pub bytes_received: u64,
    /// Bytes sent.
    pub bytes_sent: u64,
    /// Datagrams dropped as malformed.
    pub dropped_malformed: u64,
    /// Datagrams that failed SRTP authentication.
    pub dropped_unauthenticated: u64,
    /// Datagrams dropped as SRTP replays.
    pub dropped_replayed: u64,
    /// Datagrams from an untrusted source.
    pub dropped_source: u64,
    /// Datagrams of another payload type.
    pub dropped_payload: u64,
    /// Times the remote SSRC changed.
    pub ssrc_changes: u64,
    /// The address media is coming from, once learned.
    pub latched: Option<SocketAddr>,
    /// The jitter buffer's own counters.
    pub jitter: JitterStats,
}

/// A digit being sent: its event code, the event's RTP timestamp, and how
/// many packets have gone out.
#[derive(Debug, Clone, Copy)]
struct DtmfSend {
    event: u8,
    timestamp: u32,
    sent: u16,
}

/// The RFC 4733 end-of-event bit, in the second payload byte.
const END_BIT: u8 = 0x80;
/// The volume DTMF is sent at: -10 dBm0 (RFC 4733 §2.3.4).
const DTMF_VOLUME: u8 = 0x0A;

/// The RFC 4733 event code for a DTMF digit.
fn dtmf_event_code(digit: char) -> Option<u8> {
    match digit.to_ascii_uppercase() {
        d @ '0'..='9' => Some(d as u8 - b'0'),
        '*' => Some(10),
        '#' => Some(11),
        d @ 'A'..='D' => Some(d as u8 - b'A' + 12),
        _ => None,
    }
}

/// The digit for an RFC 4733 event code, for DTMF events (0-15).
fn dtmf_digit(event: u8) -> Option<char> {
    match event {
        0..=9 => Some(char::from(b'0' + event)),
        10 => Some('*'),
        11 => Some('#'),
        12..=15 => Some(char::from(b'A' + event - 12)),
        _ => None,
    }
}

/// The packet handling of one stream, without I/O or a clock of its own.
pub struct StreamCore {
    config: StreamConfig,
    srtp: Option<SrtpPair>,
    encoder: Box<dyn AudioEncoder>,
    decoder: Box<dyn AudioDecoder>,
    jitter: JitterBuffer,
    stats: StreamStats,
    remote_ssrc: Option<u32>,
    last_dtmf_timestamp: Option<u32>,
    started: Option<Instant>,
    last_packet: Option<Instant>,
    last_timeout: Option<Instant>,
    playing: bool,
    ssrc: u32,
    seq: u16,
    timestamp: u32,
    marker: bool,
    dtmf: VecDeque<u8>,
    dtmf_sending: Option<DtmfSend>,
}

impl std::fmt::Debug for StreamCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamCore")
            .field("codec", &self.config.codec)
            .field("remote", &self.config.remote)
            .field("latched", &self.stats.latched)
            .field("srtp", &self.srtp.is_some())
            .finish_non_exhaustive()
    }
}

impl StreamCore {
    /// Set up a stream. SRTP is used in both directions when `srtp` is given.
    pub fn new(config: StreamConfig, srtp: Option<SrtpPair>) -> Result<Self> {
        let jitter = JitterBuffer::new(JitterConfig {
            mode: config.jitter,
            clock_rate: config.codec.clock_rate(),
            max_packets: config.max_packets,
        });
        Ok(Self {
            encoder: create_encoder(config.codec)?,
            decoder: create_decoder(config.codec)?,
            jitter,
            srtp,
            stats: StreamStats::default(),
            remote_ssrc: None,
            last_dtmf_timestamp: None,
            started: None,
            last_packet: None,
            last_timeout: None,
            playing: false,
            ssrc: rand::random(),
            seq: rand::random(),
            timestamp: rand::random(),
            marker: true,
            dtmf: VecDeque::new(),
            dtmf_sending: None,
            config,
        })
    }

    /// Where outgoing packets go: the learned source once there is one,
    /// otherwise the SDP address.
    #[must_use]
    pub fn destination(&self) -> SocketAddr {
        self.stats.latched.unwrap_or(self.config.remote)
    }

    /// Counters so far.
    #[must_use]
    pub fn stats(&self) -> StreamStats {
        let mut stats = self.stats.clone();
        stats.jitter = self.jitter.stats();
        stats
    }

    /// The SSRC this stream sends with.
    #[must_use]
    pub fn ssrc(&self) -> u32 {
        self.ssrc
    }

    /// Take one incoming datagram from `from`, arrived at `now`.
    ///
    /// Audio goes to the jitter buffer, to come out of [`Self::playout`].
    /// A telephone-event packet that starts a new digit returns it. Anything
    /// else is dropped, counted, and changes no state.
    pub fn ingest(
        &mut self,
        datagram: &[u8],
        from: SocketAddr,
        now: Instant,
    ) -> std::result::Result<Option<DtmfEvent>, Dropped> {
        self.started.get_or_insert(now);
        let result = self.admit(datagram, from, now);
        match result {
            Err(Dropped::Malformed) => self.stats.dropped_malformed += 1,
            Err(Dropped::Unauthenticated) => self.stats.dropped_unauthenticated += 1,
            Err(Dropped::Replayed) => self.stats.dropped_replayed += 1,
            Err(Dropped::UnexpectedSource) => self.stats.dropped_source += 1,
            Err(Dropped::OtherPayload) => self.stats.dropped_payload += 1,
            Ok(_) => {}
        }
        result
    }

    fn admit(
        &mut self,
        datagram: &[u8],
        from: SocketAddr,
        now: Instant,
    ) -> std::result::Result<Option<DtmfEvent>, Dropped> {
        if datagram.len() < 12 || datagram[0] >> 6 != 2 {
            return Err(Dropped::Malformed);
        }
        // RTCP multiplexed on the RTP port (RFC 5761): packet types 200-204
        // land on 72-76 in the RTP payload type field.
        if (72..=76).contains(&(datagram[1] & 0x7F)) {
            return Err(Dropped::OtherPayload);
        }
        match self.stats.latched {
            Some(latched) if latched != from => return Err(Dropped::UnexpectedSource),
            None if self.srtp.is_none() && from.ip() != self.config.remote.ip() => {
                return Err(Dropped::UnexpectedSource);
            }
            _ => {}
        }

        let clear;
        let packet = match self.srtp.as_mut() {
            Some(srtp) => {
                clear = srtp.inbound.unprotect_rtp(datagram).map_err(|e| match e {
                    Error::SrtpReplay => Dropped::Replayed,
                    _ => Dropped::Unauthenticated,
                })?;
                clear.as_slice()
            }
            None => datagram,
        };

        let header = RtpHeader::parse(packet).ok_or(Dropped::Malformed)?;
        let start = header.header_length();
        let mut end = packet.len();
        if header.padding {
            let pad = usize::from(*packet.last().ok_or(Dropped::Malformed)?);
            if pad == 0 || pad > end - start {
                return Err(Dropped::Malformed);
            }
            end -= pad;
        }
        let payload = packet.get(start..end).ok_or(Dropped::Malformed)?;

        let is_audio = header.payload_type == self.config.payload_type;
        let is_dtmf = Some(header.payload_type) == self.config.dtmf_payload_type;
        if !is_audio && !is_dtmf {
            return Err(Dropped::OtherPayload);
        }
        if is_dtmf && payload.len() < 4 {
            return Err(Dropped::Malformed);
        }

        // Trusted from here on.
        if self.stats.latched.is_none() {
            log::debug!("Media stream: latched to {from}");
            self.stats.latched = Some(from);
        }
        if self.remote_ssrc.is_some_and(|ssrc| ssrc != header.ssrc) {
            log::debug!("Media stream: remote SSRC changed to {}", header.ssrc);
            self.jitter.reset();
            self.decoder = create_decoder(self.config.codec).map_err(|_| Dropped::Malformed)?;
            self.last_dtmf_timestamp = None;
            self.playing = false;
            self.stats.ssrc_changes += 1;
        }
        self.remote_ssrc = Some(header.ssrc);
        self.last_packet = Some(now);
        self.stats.packets_received += 1;
        self.stats.bytes_received += datagram.len() as u64;

        if is_dtmf {
            // One press is one event timestamp; its updates and the three
            // end packets repeat it (RFC 4733 §2.5.1).
            if self.last_dtmf_timestamp == Some(header.timestamp) {
                return Ok(None);
            }
            let Some(digit) = dtmf_digit(payload[0]) else {
                return Ok(None);
            };
            self.last_dtmf_timestamp = Some(header.timestamp);
            return Ok(Some(DtmfEvent {
                digit,
                event: payload[0],
            }));
        }

        self.jitter
            .push_at(header.sequence, header.timestamp, payload.to_vec(), now);
        Ok(None)
    }

    /// The events due at a 20 ms playout tick at `now`: a [`StreamEvent::Timeout`]
    /// when RTP has been missing long enough, and once audio has started, the
    /// next frame.
    pub fn playout(&mut self, now: Instant) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let start = *self.started.get_or_insert(now);
        let since = self.last_packet.unwrap_or(start);
        let reported = self.last_timeout.map_or(since, |t| t.max(since));
        let silent_for = now.saturating_duration_since(since);
        if now.saturating_duration_since(reported) >= self.config.rtp_timeout {
            self.last_timeout = Some(now);
            events.push(StreamEvent::Timeout { silent_for });
        }

        let samples = self.config.codec.pcm_samples_per_packet();
        if let Some(packet) = self.jitter.pop_at(now) {
            self.playing = true;
            let mut pcm = Vec::with_capacity(samples);
            let kind = if packet.payload.is_empty() {
                let next = self.jitter.peek_next().map(|p| p.payload.as_slice());
                self.decoder.conceal_loss(samples, next, &mut pcm);
                FrameKind::Concealed
            } else {
                self.decoder.decode(&packet.payload, &mut pcm);
                FrameKind::Received
            };
            // A peer sending a different ptime, or a corrupt payload, must not
            // change the frame size the caller relies on.
            pcm.resize(samples, 0);
            events.push(StreamEvent::Frame(AudioFrame { pcm, kind }));
        } else if self.playing {
            events.push(StreamEvent::Frame(AudioFrame {
                pcm: vec![0; samples],
                kind: FrameKind::Silence,
            }));
        }
        events
    }

    /// Queue a DTMF digit to be sent as RFC 4733 events in place of the next
    /// audio frames.
    pub fn queue_dtmf(&mut self, digit: char) -> Result<()> {
        let event = dtmf_event_code(digit)
            .ok_or_else(|| Error::invalid_parameter(format!("not a DTMF digit: {digit:?}")))?;
        if self.config.dtmf_payload_type.is_none() {
            return Err(Error::invalid_parameter(
                "no telephone-event payload type was negotiated",
            ));
        }
        self.dtmf.push_back(event);
        Ok(())
    }

    /// Build the datagrams for the next 20 ms: the audio `pcm` encoded, or,
    /// while a digit is being sent, its telephone-event packet instead (the
    /// last one three times, RFC 4733 §2.5.1.4). `pcm` must be exactly one
    /// packet's worth at the codec's rate.
    pub fn packetize(&mut self, pcm: &[i16]) -> Result<Vec<Vec<u8>>> {
        let samples = self.config.codec.pcm_samples_per_packet();
        if pcm.len() != samples {
            return Err(Error::invalid_parameter(format!(
                "a frame is {samples} samples for {}, got {}",
                self.config.codec,
                pcm.len()
            )));
        }
        let ticks = self.config.codec.rtp_ticks_per_packet();
        let mut datagrams = Vec::new();

        if self.dtmf_sending.is_none()
            && let Some(event) = self.dtmf.pop_front()
        {
            self.dtmf_sending = Some(DtmfSend {
                event,
                timestamp: self.dtmf_timestamp(),
                sent: 0,
            });
        }

        if let (Some(mut send), Some(pt)) = (self.dtmf_sending, self.config.dtmf_payload_type) {
            send.sent += 1;
            let per_packet = self.config.dtmf_clock_rate * CodecType::PACKET_DURATION_MS / 1000;
            let duration = (u32::from(send.sent) * per_packet).min(u32::from(u16::MAX)) as u16;
            let end = send.sent >= self.config.dtmf_packets.max(1);
            let flags = if end {
                END_BIT | DTMF_VOLUME
            } else {
                DTMF_VOLUME
            };
            let mut payload = vec![send.event, flags];
            payload.extend_from_slice(&duration.to_be_bytes());
            for _ in 0..if end { 3 } else { 1 } {
                let mut header = RtpHeader::new(pt, self.seq, send.timestamp, self.ssrc);
                if send.sent == 1 {
                    header = header.with_marker();
                }
                datagrams.push(self.finish(header, &payload)?);
            }
            self.dtmf_sending = (!end).then_some(send);
            // Audio after the digit starts a new talkspurt.
            self.marker = true;
        } else {
            let mut payload = Vec::with_capacity(samples);
            self.encoder.encode(pcm, &mut payload);
            let mut header = RtpHeader::new(
                self.config.payload_type,
                self.seq,
                self.timestamp,
                self.ssrc,
            );
            if self.marker {
                header = header.with_marker();
                self.marker = false;
            }
            datagrams.push(self.finish(header, &payload)?);
        }
        self.timestamp = self.timestamp.wrapping_add(ticks);
        Ok(datagrams)
    }

    /// The current audio timestamp on the telephone-event clock.
    fn dtmf_timestamp(&self) -> u32 {
        let audio = self.config.codec.clock_rate();
        let event = self.config.dtmf_clock_rate;
        if audio == event || audio == 0 {
            self.timestamp
        } else {
            (u64::from(self.timestamp) * u64::from(event) / u64::from(audio)) as u32
        }
    }

    /// Serialize, protect and count one packet, advancing the sequence.
    fn finish(&mut self, header: RtpHeader, payload: &[u8]) -> Result<Vec<u8>> {
        let mut packet = header.to_bytes();
        packet.extend_from_slice(payload);
        self.seq = self.seq.wrapping_add(1);
        let packet = match self.srtp.as_mut() {
            Some(srtp) => srtp.outbound.protect_rtp(&packet)?,
            None => packet,
        };
        self.stats.packets_sent += 1;
        self.stats.bytes_sent += packet.len() as u64;
        Ok(packet)
    }

    /// An RTCP sender report for the stream so far, SRTCP-protected when
    /// keyed.
    pub fn sender_report(&mut self) -> Result<Vec<u8>> {
        let report = build_rtcp_sr(
            self.ssrc,
            self.stats.packets_sent as u32,
            self.stats.bytes_sent as u32,
            self.timestamp,
        );
        match self.srtp.as_mut() {
            Some(srtp) => srtp.outbound.protect_rtcp(&report),
            None => Ok(report),
        }
    }
}

/// Whether a receive error is the ICMP echo of an earlier send to a closed
/// port, which some platforms report on the next receive. Not fatal: the peer
/// may simply not be listening yet.
fn is_transient(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionRefused
    )
}

/// A headless media stream over a UDP socket the caller provides.
///
/// Runs entirely on the task that drives it: [`Self::recv_frame`] and
/// [`Self::send_frame`] are meant to be called from one `select!` loop.
pub struct MediaStream {
    socket: UdpSocket,
    rtcp: Option<UdpSocket>,
    core: StreamCore,
    ticker: tokio::time::Interval,
    pending: VecDeque<StreamEvent>,
    buf: Vec<u8>,
}

impl std::fmt::Debug for MediaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaStream")
            .field("local", &self.socket.local_addr().ok())
            .field("core", &self.core)
            .finish_non_exhaustive()
    }
}

impl MediaStream {
    /// A stream on `socket` (already bound to the local RTP port), configured
    /// from the negotiated SDP. SRTP is used when `srtp` is given.
    pub fn new(socket: UdpSocket, config: StreamConfig, srtp: Option<SrtpPair>) -> Result<Self> {
        let mut ticker = tokio::time::interval(Duration::from_millis(u64::from(
            CodecType::PACKET_DURATION_MS,
        )));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        Ok(Self {
            socket,
            rtcp: None,
            core: StreamCore::new(config, srtp)?,
            ticker,
            pending: VecDeque::new(),
            buf: vec![0; 2048],
        })
    }

    /// Send RTCP from `rtcp` (bound to the RTP port + 1) on [`Self::rtcp_tick`].
    #[must_use]
    pub fn with_rtcp(mut self, rtcp: UdpSocket) -> Self {
        self.rtcp = Some(rtcp);
        self
    }

    /// The local RTP address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Wait for the next event: a frame every 20 ms once audio has started, a
    /// digit as soon as it arrives, or a timeout report.
    ///
    /// Cancel-safe: dropping the future loses nothing already received.
    pub async fn recv_frame(&mut self) -> Result<StreamEvent> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(event);
            }
            tokio::select! {
                received = self.socket.recv_from(&mut self.buf) => match received {
                    Ok((len, from)) => {
                        let now = tokio::time::Instant::now().into_std();
                        if let Ok(Some(digit)) = self.core.ingest(&self.buf[..len], from, now) {
                            return Ok(StreamEvent::Dtmf(digit));
                        }
                    }
                    Err(e) if is_transient(&e) => {}
                    Err(e) => return Err(e.into()),
                },
                _ = self.ticker.tick() => {
                    let now = tokio::time::Instant::now().into_std();
                    self.pending.extend(self.core.playout(now));
                }
            }
        }
    }

    /// Encode and send 20 ms of PCM at the codec's rate, or the next packet of
    /// a queued digit in its place.
    pub async fn send_frame(&mut self, pcm: &[i16]) -> Result<()> {
        let destination = self.core.destination();
        for datagram in self.core.packetize(pcm)? {
            self.socket.send_to(&datagram, destination).await?;
        }
        Ok(())
    }

    /// Queue a DTMF digit (`0`-`9`, `*`, `#`, `A`-`D`). It goes out as RFC 4733
    /// events over the next few [`Self::send_frame`] calls, replacing their
    /// audio.
    pub fn send_dtmf(&mut self, digit: char) -> Result<()> {
        self.core.queue_dtmf(digit)
    }

    /// Send an RTCP sender report, if an RTCP socket was given, to the RTP
    /// destination's port + 1; and discard whatever RTCP has arrived.
    pub async fn rtcp_tick(&mut self) -> Result<()> {
        let Some(rtcp) = self.rtcp.as_ref() else {
            return Ok(());
        };
        let mut scratch = [0u8; 1500];
        while rtcp.try_recv_from(&mut scratch).is_ok() {}
        let rtp = self.core.destination();
        let destination = SocketAddr::new(rtp.ip(), rtp.port().wrapping_add(1));
        let report = self.core.sender_report()?;
        match rtcp.send_to(&report, destination).await {
            Err(e) if !is_transient(&e) => Err(e.into()),
            _ => Ok(()),
        }
    }

    /// Counters so far.
    #[must_use]
    pub fn stats(&self) -> StreamStats {
        self.core.stats()
    }

    /// The packet handling underneath, for callers that need it directly.
    pub fn core_mut(&mut self) -> &mut StreamCore {
        &mut self.core
    }
}

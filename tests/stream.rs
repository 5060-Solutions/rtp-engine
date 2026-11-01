//! The headless per-call media stream: real sockets for the round trips,
//! the sans-IO core for everything that needs exact control.
#![cfg(feature = "stream")]

use std::net::{Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};

use rtp_engine::CodecType;
use rtp_engine::jitter::JitterMode;
use rtp_engine::rtp::RtpHeader;
use rtp_engine::srtp::SrtpContext;
use rtp_engine::stream::{
    Dropped, FrameKind, MediaStream, SrtpPair, StreamConfig, StreamCore, StreamEvent,
};
use tokio::net::UdpSocket;

const PEER: SocketAddr = SocketAddr::new(std::net::IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)), 4000);
const ELSEWHERE: SocketAddr =
    SocketAddr::new(std::net::IpAddr::V4(Ipv4Addr::new(10, 9, 9, 9)), 4000);

fn tone(samples: usize, rate: u32, hz: f32, start: usize) -> Vec<i16> {
    (start..start + samples)
        .map(|n| {
            let t = n as f32 / rate as f32;
            (8000.0 * (2.0 * std::f32::consts::PI * hz * t).sin()) as i16
        })
        .collect()
}

fn rtp(pt: u8, seq: u16, ts: u32, ssrc: u32, payload: &[u8]) -> Vec<u8> {
    let mut packet = RtpHeader::new(pt, seq, ts, ssrc).to_bytes();
    packet.extend_from_slice(payload);
    packet
}

/// A pair of SRTP directions with fresh keys, as SDES would give both ends.
fn srtp_pairs() -> (SrtpPair, SrtpPair) {
    let (a_out, a_key) = SrtpContext::generate().unwrap();
    let (b_out, b_key) = SrtpContext::generate().unwrap();
    let a = SrtpPair {
        inbound: SrtpContext::from_base64(&b_key).unwrap(),
        outbound: a_out,
    };
    let b = SrtpPair {
        inbound: SrtpContext::from_base64(&a_key).unwrap(),
        outbound: b_out,
    };
    (a, b)
}

async fn socket() -> UdpSocket {
    UdpSocket::bind(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0))
        .await
        .unwrap()
}

/// Two streams over loopback UDP with SRTP: what one sends, the other
/// decodes, in order, as audio rather than concealment.
async fn round_trip(codec: CodecType, srtp: bool) {
    let (a_socket, b_socket) = (socket().await, socket().await);
    let (a_addr, b_addr) = (
        a_socket.local_addr().unwrap(),
        b_socket.local_addr().unwrap(),
    );
    let (a_keys, b_keys) = if srtp {
        let (a, b) = srtp_pairs();
        (Some(a), Some(b))
    } else {
        (None, None)
    };
    let mut a = MediaStream::new(a_socket, StreamConfig::new(codec, b_addr), a_keys).unwrap();
    let mut b = MediaStream::new(b_socket, StreamConfig::new(codec, a_addr), b_keys).unwrap();

    let samples = codec.pcm_samples_per_packet();
    let sender = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(20));
        for n in 0..50 {
            ticker.tick().await;
            a.send_frame(&tone(samples, codec.pcm_rate(), 440.0, n * samples))
                .await
                .unwrap();
        }
        a
    });

    let mut received = 0;
    let mut energy = 0.0_f64;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while received < 40 && tokio::time::Instant::now() < deadline {
        let event = tokio::time::timeout(Duration::from_secs(1), b.recv_frame())
            .await
            .unwrap()
            .unwrap();
        if let StreamEvent::Frame(frame) = event {
            assert_eq!(frame.pcm.len(), samples, "20 ms at the codec's PCM rate");
            if frame.kind == FrameKind::Received {
                received += 1;
                energy += frame.pcm.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>();
            }
        }
    }
    let a = sender.await.unwrap();
    assert!(received >= 40, "only {received} frames arrived");
    assert!(energy > 0.0, "the tone came through");
    assert_eq!(a.stats().packets_sent, 50);
    let stats = b.stats();
    assert_eq!(stats.dropped_unauthenticated, 0);
    assert_eq!(stats.latched, Some(a_addr));
}

#[tokio::test]
async fn stream_round_trip_pcmu_srtp() {
    round_trip(CodecType::Pcmu, true).await;
}

#[tokio::test]
async fn stream_round_trip_g722() {
    round_trip(CodecType::G722, false).await;
}

#[cfg(feature = "opus")]
#[tokio::test]
async fn stream_round_trip_opus_srtp() {
    round_trip(CodecType::Opus, true).await;
}

/// A digit queued on one side arrives on the other once, over SRTP, and
/// audio resumes after it.
#[tokio::test]
async fn dtmf_zero_round_trip() {
    let (a_socket, b_socket) = (socket().await, socket().await);
    let (a_addr, b_addr) = (
        a_socket.local_addr().unwrap(),
        b_socket.local_addr().unwrap(),
    );
    let (a_keys, b_keys) = srtp_pairs();
    let codec = CodecType::Pcmu;
    let mut a = MediaStream::new(a_socket, StreamConfig::new(codec, b_addr), Some(a_keys)).unwrap();
    let mut b = MediaStream::new(b_socket, StreamConfig::new(codec, a_addr), Some(b_keys)).unwrap();

    a.send_dtmf('0').unwrap();
    for n in 0..15 {
        a.send_frame(&tone(160, 8000, 300.0, n * 160))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let mut digits = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_millis(800);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), b.recv_frame()).await {
            Ok(Ok(StreamEvent::Dtmf(event))) => digits.push(event.digit),
            Ok(Ok(_)) | Err(_) => {}
            Ok(Err(e)) => panic!("{e}"),
        }
    }
    assert_eq!(digits, vec!['0']);
}

#[test]
fn send_dtmf_refuses_a_non_digit() {
    let mut core = StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), None).unwrap();
    assert!(core.queue_dtmf('x').is_err());
    for digit in ['0', '9', '*', '#', 'A', 'd'] {
        assert!(core.queue_dtmf(digit).is_ok(), "{digit}");
    }
}

/// RFC 4733 §2.5.1.4: the end packet is sent three times, and every packet
/// of one event shares its timestamp. One press is one digit.
#[test]
fn rfc4733_event_decoded_once_with_end_bit_repeats() {
    let mut core = StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), None).unwrap();
    let now = Instant::now();
    let mut digits = Vec::new();
    for (seq, (end, duration)) in [
        (false, 160u16),
        (false, 320),
        (true, 480),
        (true, 480),
        (true, 480),
    ]
    .into_iter()
    .enumerate()
    {
        let flags = if end { 0x8A } else { 10 };
        let payload = [0u8, flags, (duration >> 8) as u8, duration as u8];
        let packet = rtp(101, 100 + seq as u16, 8000, 7, &payload);
        if let Ok(Some(event)) = core.ingest(&packet, PEER, now) {
            digits.push(event.digit);
        }
    }
    // A second press of the same digit is a new event (a new timestamp).
    let packet = rtp(101, 105, 9600, 7, &[0, 0x8A, 0, 160]);
    if let Ok(Some(event)) = core.ingest(&packet, PEER, now) {
        digits.push(event.digit);
    }
    assert_eq!(digits, vec!['0', '0']);
}

/// A telephone-event payload type equal to the audio one would turn every
/// audio packet into a digit: refused.
#[test]
fn dtmf_on_the_audio_payload_type_is_refused() {
    let mut config = StreamConfig::new(CodecType::Pcmu, PEER);
    config.dtmf_payload_type = Some(0);
    assert!(StreamCore::new(config, None).is_err());
}

/// A late end packet of an earlier digit is not that digit pressed again:
/// events older than the newest one seen are ignored (RFC 4733 timestamps
/// only move forward within a stream).
#[test]
fn a_stale_event_is_not_a_new_press() {
    let mut core = StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), None).unwrap();
    let now = Instant::now();
    let event = |seq: u16, ts: u32, digit: u8, end: bool| {
        let flags = if end { 0x8A } else { 0x0A };
        rtp(101, seq, ts, 7, &[digit, flags, 0x01, 0xE0])
    };
    let mut digits = Vec::new();
    for packet in [
        event(1, 1000, 1, false),
        event(2, 1000, 1, true),
        event(3, 2000, 2, false),
        event(4, 1000, 1, true),
    ] {
        if let Ok(Some(e)) = core.ingest(&packet, PEER, now) {
            digits.push(e.digit);
        }
    }
    assert_eq!(digits, vec!['1', '2']);

    // Timestamps wrap: an event just past u32::MAX is newer, not older.
    let mut core = StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), None).unwrap();
    let mut digits = Vec::new();
    for packet in [event(5, u32::MAX - 100, 3, false), event(6, 50, 4, false)] {
        if let Ok(Some(e)) = core.ingest(&packet, PEER, now) {
            digits.push(e.digit);
        }
    }
    assert_eq!(digits, vec!['3', '4']);
}

/// Packets that fail SRTP authentication are dropped and counted, never
/// played, and never latch the stream to their source.
#[test]
fn srtp_auth_failure_packets_dropped() {
    let (_, b_keys) = srtp_pairs();
    let mut core = StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), Some(b_keys)).unwrap();
    let (mut forger, _) = SrtpContext::generate().unwrap();
    let forged = forger
        .protect_rtp(&rtp(0, 1, 160, 9, &[0xFF; 160]))
        .unwrap();
    assert_eq!(
        core.ingest(&forged, ELSEWHERE, Instant::now()),
        Err(Dropped::Unauthenticated)
    );
    let stats = core.stats();
    assert_eq!(stats.dropped_unauthenticated, 1);
    assert_eq!(stats.latched, None);
    assert_eq!(stats.jitter.packets_received, 0);
}

#[test]
fn srtp_replayed_packet_dropped() {
    let (a_keys, b_keys) = srtp_pairs();
    let mut sender = a_keys.outbound;
    let mut core = StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), Some(b_keys)).unwrap();
    let packet = sender
        .protect_rtp(&rtp(0, 1, 160, 9, &[0xFF; 160]))
        .unwrap();
    let now = Instant::now();
    assert_eq!(core.ingest(&packet, PEER, now), Ok(None));
    assert_eq!(core.ingest(&packet, PEER, now), Err(Dropped::Replayed));
    assert_eq!(core.stats().dropped_replayed, 1);
}

/// Symmetric RTP: the stream answers where the media really comes from, but
/// only learns that from a packet it can trust: plain RTP from the host the
/// SDP named, or any packet that authenticated under SRTP. Once latched,
/// nothing else gets in.
#[test]
fn rtp_from_unexpected_source_ignored_after_latch() {
    let mut core = StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), None).unwrap();
    let now = Instant::now();
    let nat_port = SocketAddr::new(PEER.ip(), 31_000);
    assert_eq!(
        core.ingest(&rtp(0, 1, 160, 9, &[0; 160]), nat_port, now),
        Ok(None)
    );
    assert_eq!(core.stats().latched, Some(nat_port));
    assert_eq!(
        core.ingest(&rtp(0, 2, 320, 9, &[0; 160]), PEER, now),
        Err(Dropped::UnexpectedSource),
        "once latched, even the SDP address is someone else"
    );
    assert_eq!(core.destination(), nat_port);
}

#[test]
fn latch_only_to_sdp_host_or_first_authenticated_srtp() {
    let now = Instant::now();
    let mut plain = StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), None).unwrap();
    assert_eq!(
        plain.ingest(&rtp(0, 1, 160, 9, &[0; 160]), ELSEWHERE, now),
        Err(Dropped::UnexpectedSource),
        "plain RTP from a host the SDP did not name"
    );
    assert_eq!(plain.stats().latched, None);
    assert_eq!(plain.destination(), PEER);

    let (a_keys, b_keys) = srtp_pairs();
    let mut sender = a_keys.outbound;
    let mut secure =
        StreamCore::new(StreamConfig::new(CodecType::Pcmu, PEER), Some(b_keys)).unwrap();
    let packet = sender.protect_rtp(&rtp(0, 1, 160, 9, &[0; 160])).unwrap();
    assert_eq!(secure.ingest(&packet, ELSEWHERE, now), Ok(None));
    assert_eq!(
        secure.stats().latched,
        Some(ELSEWHERE),
        "authenticated, so trusted"
    );
}

/// A new SSRC is a new stream: the old one's buffered packets and sequence
/// numbers must not hold it up or be mixed into it.
#[test]
fn ssrc_change_resets_the_buffer() {
    let mut config = StreamConfig::new(CodecType::Pcmu, PEER);
    config.jitter = JitterMode::Fixed { delay_ms: 0 };
    let mut core = StreamCore::new(config, None).unwrap();
    let t0 = Instant::now();
    for seq in 0..5u16 {
        core.ingest(
            &rtp(0, seq, u32::from(seq) * 160, 1, &[0x10; 160]),
            PEER,
            t0,
        )
        .unwrap();
    }
    core.ingest(&rtp(0, 40_000, 7, 2, &[0x20; 160]), PEER, t0)
        .unwrap();
    assert_eq!(core.stats().ssrc_changes, 1);
    let events = core.playout(t0 + Duration::from_millis(20));
    let frame = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::Frame(f) => Some(f),
            _ => None,
        })
        .unwrap();
    assert_eq!(frame.kind, FrameKind::Received);
    let mut expected = Vec::new();
    rtp_engine::codec::create_decoder(CodecType::Pcmu)
        .unwrap()
        .decode(&[0x20; 160], &mut expected);
    assert_eq!(frame.pcm, expected, "the new source's audio, not the old");
}

/// No RTP at all is reported, repeatedly, so the caller can hang up a dead
/// call; and a pause in a live stream plays as silence at the same cadence.
#[test]
fn silence_and_timeouts_are_reported() {
    let mut config = StreamConfig::new(CodecType::Pcmu, PEER);
    config.rtp_timeout = Duration::from_millis(500);
    let mut core = StreamCore::new(config, None).unwrap();
    let t0 = Instant::now();
    // The clock starts at the stream's first tick.
    assert_eq!(core.playout(t0), vec![]);
    assert_eq!(core.playout(t0 + Duration::from_millis(100)), vec![]);
    let timeouts = |events: &[StreamEvent]| {
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Timeout { .. }))
            .count()
    };
    assert_eq!(timeouts(&core.playout(t0 + Duration::from_millis(500))), 1);
    assert_eq!(timeouts(&core.playout(t0 + Duration::from_millis(520))), 0);
    assert_eq!(timeouts(&core.playout(t0 + Duration::from_millis(1000))), 1);

    let t1 = t0 + Duration::from_secs(2);
    core.ingest(&rtp(0, 1, 160, 3, &[0xFF; 160]), PEER, t1)
        .unwrap();
    let kinds: Vec<FrameKind> = (1..10)
        .flat_map(|n| core.playout(t1 + Duration::from_millis(40 + n * 20)))
        .filter_map(|e| match e {
            StreamEvent::Frame(f) => Some(f.kind),
            _ => None,
        })
        .collect();
    assert_eq!(kinds[0], FrameKind::Received);
    assert!(
        kinds[1..].iter().all(|k| *k == FrameKind::Silence),
        "{kinds:?}"
    );
}

/// The agent's jitter targets are the defaults: target 40, min 20, max 120.
#[test]
fn jitter_targets_40_20_120() {
    let config = StreamConfig::new(CodecType::Pcmu, PEER);
    assert_eq!(
        config.jitter,
        JitterMode::Adaptive {
            target_ms: 40,
            min_ms: 20,
            max_ms: 120
        }
    );
    let core = StreamCore::new(config, None).unwrap();
    assert_eq!(core.stats().jitter.current_delay_ms, 40);
}

/// Outgoing packets carry the negotiated payload type, a continuous
/// sequence and the codec's RTP clock, SRTP-protected when keyed.
#[test]
fn packetize_numbers_and_clocks_the_stream() {
    let mut config = StreamConfig::new(CodecType::G722, PEER);
    config.payload_type = 9;
    let mut core = StreamCore::new(config, None).unwrap();
    let first = core.packetize(&[0; 320]).unwrap().remove(0);
    let second = core.packetize(&[0; 320]).unwrap().remove(0);
    let (h1, h2) = (
        RtpHeader::parse(&first).unwrap(),
        RtpHeader::parse(&second).unwrap(),
    );
    assert_eq!(h1.payload_type, 9);
    assert!(h1.marker, "the first packet of a talkspurt is marked");
    assert_eq!(h2.sequence, h1.sequence.wrapping_add(1));
    assert_eq!(
        h2.timestamp,
        h1.timestamp.wrapping_add(160),
        "8000 Hz clock"
    );
    assert_eq!(first.len(), 12 + 160);
    assert!(core.packetize(&[0; 100]).is_err(), "a frame must be 20 ms");
}

mod fuzz {
    use super::*;
    use proptest::prelude::*;

    /// The codecs the fuzzers drive: every decoder sees arbitrary payloads.
    fn codecs() -> Vec<CodecType> {
        let mut codecs = vec![CodecType::Pcmu, CodecType::Pcma, CodecType::G722];
        #[cfg(feature = "opus")]
        codecs.push(CodecType::Opus);
        codecs
    }

    fn core(srtp: bool, codec: usize) -> StreamCore {
        let keys = srtp.then(|| srtp_pairs().1);
        let codecs = codecs();
        let codec = codecs[codec % codecs.len()];
        StreamCore::new(StreamConfig::new(codec, PEER), keys).unwrap()
    }

    fn core_payload_type(codec: usize) -> u8 {
        let codecs = codecs();
        codecs[codec % codecs.len()].payload_type()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        /// Arbitrary datagrams never panic the stream, plain or SRTP, and
        /// playout carries on afterwards.
        #[test]
        fn arbitrary_datagrams_never_panic(
            datagrams in proptest::collection::vec(proptest::collection::vec(any::<u8>(), 0..300), 1..20),
            srtp in any::<bool>(),
            codec in any::<usize>(),
        ) {
            let mut core = core(srtp, codec);
            let t0 = Instant::now();
            for (n, datagram) in datagrams.iter().enumerate() {
                let _ = core.ingest(datagram, PEER, t0 + Duration::from_millis(n as u64 * 20));
                let _ = core.playout(t0 + Duration::from_millis(n as u64 * 20 + 10));
            }
            prop_assert!(core.stats().jitter.buffer_depth <= 50);
        }

        /// Valid packets with mutated headers (version, CSRC count,
        /// extension and padding bits, lengths) never panic either.
        #[test]
        fn mutated_rtp_headers_never_panic(
            first in any::<u8>(),
            pt in any::<u8>(),
            seq in any::<u16>(),
            payload in proptest::collection::vec(any::<u8>(), 0..200),
            cut in 0usize..240,
            codec in any::<usize>(),
        ) {
            let mut core = core(false, codec);
            // Mostly on the audio payload type, so the decoder runs.
            let pt = if pt % 4 == 0 { pt & 0x7F } else { core_payload_type(codec) };
            let mut packet = rtp(pt, seq, 0, 1, &payload);
            packet[0] = 0x80 | (first & 0x3F);
            packet.truncate(cut.max(1));
            let now = Instant::now();
            let _ = core.ingest(&packet, PEER, now);
            let _ = core.playout(now + Duration::from_millis(100));
        }
    }
}

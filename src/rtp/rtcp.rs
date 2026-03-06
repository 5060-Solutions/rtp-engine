//! RTCP (RTP Control Protocol) implementation.
//!
//! Provides Sender Reports (SR) and Receiver Reports (RR) per RFC 3550.

/// RTCP packet types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RtcpType {
    /// Sender Report
    SenderReport = 200,
    /// Receiver Report
    ReceiverReport = 201,
    /// Source Description
    SourceDescription = 202,
    /// Goodbye
    Goodbye = 203,
    /// Application-defined
    ApplicationDefined = 204,
}

impl RtcpType {
    /// Parse an RTCP type from a byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            200 => Some(Self::SenderReport),
            201 => Some(Self::ReceiverReport),
            202 => Some(Self::SourceDescription),
            203 => Some(Self::Goodbye),
            204 => Some(Self::ApplicationDefined),
            _ => None,
        }
    }
}

/// A parsed RTCP packet.
#[derive(Debug, Clone)]
pub struct RtcpPacket {
    /// Packet type.
    pub packet_type: RtcpType,
    /// SSRC of the sender.
    pub ssrc: u32,
    /// Raw packet data.
    pub data: Vec<u8>,
}

impl RtcpPacket {
    /// Parse an RTCP packet from raw bytes.
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }

        let version = (data[0] >> 6) & 0x03;
        if version != 2 {
            return None;
        }

        let packet_type = RtcpType::from_byte(data[1])?;
        let ssrc = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);

        Some(Self {
            packet_type,
            ssrc,
            data: data.to_vec(),
        })
    }
}

/// Build an RTCP Sender Report (SR) packet.
///
/// # Arguments
/// * `ssrc` - Synchronization source identifier
/// * `packets_sent` - Total RTP packets sent
/// * `octets_sent` - Total payload octets sent
pub fn build_rtcp_sr(ssrc: u32, packets_sent: u32, octets_sent: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(28);

    // V=2, P=0, RC=0, PT=200 (SR)
    buf.push(0x80);
    buf.push(200);

    // Length in 32-bit words minus one = 6
    buf.extend_from_slice(&6u16.to_be_bytes());

    // SSRC
    buf.extend_from_slice(&ssrc.to_be_bytes());

    // NTP timestamp (simplified: use system time)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let ntp_sec = now.as_secs() + 2_208_988_800; // NTP epoch offset
    let ntp_frac = ((now.subsec_nanos() as u64) << 32) / 1_000_000_000;
    buf.extend_from_slice(&(ntp_sec as u32).to_be_bytes());
    buf.extend_from_slice(&(ntp_frac as u32).to_be_bytes());

    // RTP timestamp (approximate)
    let rtp_ts = (now.as_millis() * 8) as u32; // 8kHz clock
    buf.extend_from_slice(&rtp_ts.to_be_bytes());

    // Sender's packet count
    buf.extend_from_slice(&packets_sent.to_be_bytes());

    // Sender's octet count
    buf.extend_from_slice(&octets_sent.to_be_bytes());

    buf
}

/// Build an RTCP Receiver Report (RR) packet.
///
/// # Arguments
/// * `ssrc` - Our SSRC
/// * `remote_ssrc` - SSRC of the source we're reporting on
/// * `loss_fraction` - Fraction of packets lost (0-255)
/// * `cumulative_lost` - Total packets lost (24-bit)
/// * `highest_seq` - Highest sequence number received (with cycles)
/// * `jitter` - Interarrival jitter
pub fn build_rtcp_rr(
    ssrc: u32,
    remote_ssrc: u32,
    loss_fraction: u8,
    cumulative_lost: u32,
    highest_seq: u32,
    jitter: u32,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);

    // V=2, P=0, RC=1, PT=201 (RR)
    buf.push(0x81);
    buf.push(201);

    // Length = 7 (32-bit words minus one)
    buf.extend_from_slice(&7u16.to_be_bytes());

    // Our SSRC
    buf.extend_from_slice(&ssrc.to_be_bytes());

    // Report block
    buf.extend_from_slice(&remote_ssrc.to_be_bytes());

    // Fraction lost (8 bits) + cumulative lost (24 bits)
    let lost_word = ((loss_fraction as u32) << 24) | (cumulative_lost & 0x00FF_FFFF);
    buf.extend_from_slice(&lost_word.to_be_bytes());

    // Extended highest sequence number
    buf.extend_from_slice(&highest_seq.to_be_bytes());

    // Interarrival jitter
    buf.extend_from_slice(&jitter.to_be_bytes());

    // Last SR timestamp (0 for now)
    buf.extend_from_slice(&0u32.to_be_bytes());

    // Delay since last SR (0 for now)
    buf.extend_from_slice(&0u32.to_be_bytes());

    buf
}

/// Build an RTCP BYE packet.
///
/// # Arguments
/// * `ssrc` - SSRC of the source leaving
pub fn build_rtcp_bye(ssrc: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8);

    // V=2, P=0, SC=1, PT=203 (BYE)
    buf.push(0x81);
    buf.push(203);

    // Length = 1 (one 32-bit word of SSRC)
    buf.extend_from_slice(&1u16.to_be_bytes());

    // SSRC
    buf.extend_from_slice(&ssrc.to_be_bytes());

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_rtcp_sr() {
        let sr = build_rtcp_sr(0x12345678, 100, 16000);

        // Check header
        assert_eq!(sr[0] & 0xC0, 0x80); // V=2
        assert_eq!(sr[1], 200); // PT=SR

        // Check SSRC
        let ssrc = u32::from_be_bytes([sr[4], sr[5], sr[6], sr[7]]);
        assert_eq!(ssrc, 0x12345678);
    }

    #[test]
    fn test_build_rtcp_rr() {
        let rr = build_rtcp_rr(0xAAAAAAAA, 0xBBBBBBBB, 25, 100, 5000, 160);

        assert_eq!(rr[0] & 0xC0, 0x80); // V=2
        assert_eq!(rr[0] & 0x1F, 1); // RC=1
        assert_eq!(rr[1], 201); // PT=RR

        // Check our SSRC
        let ssrc = u32::from_be_bytes([rr[4], rr[5], rr[6], rr[7]]);
        assert_eq!(ssrc, 0xAAAAAAAA);

        // Check remote SSRC
        let remote_ssrc = u32::from_be_bytes([rr[8], rr[9], rr[10], rr[11]]);
        assert_eq!(remote_ssrc, 0xBBBBBBBB);
    }

    #[test]
    fn test_build_rtcp_bye() {
        let bye = build_rtcp_bye(0xDEADBEEF);

        assert_eq!(bye.len(), 8);
        assert_eq!(bye[1], 203); // PT=BYE

        let ssrc = u32::from_be_bytes([bye[4], bye[5], bye[6], bye[7]]);
        assert_eq!(ssrc, 0xDEADBEEF);
    }

    #[test]
    fn test_parse_rtcp() {
        let sr = build_rtcp_sr(0x12345678, 100, 16000);
        let parsed = RtcpPacket::parse(&sr).unwrap();

        assert_eq!(parsed.packet_type, RtcpType::SenderReport);
        assert_eq!(parsed.ssrc, 0x12345678);
    }

    #[test]
    fn test_parse_rtcp_rr() {
        let rr = build_rtcp_rr(0xAAAAAAAA, 0xBBBBBBBB, 25, 100, 5000, 160);
        let parsed = RtcpPacket::parse(&rr).unwrap();

        assert_eq!(parsed.packet_type, RtcpType::ReceiverReport);
        assert_eq!(parsed.ssrc, 0xAAAAAAAA);
    }

    #[test]
    fn test_parse_rtcp_bye() {
        let bye = build_rtcp_bye(0xDEADBEEF);
        let parsed = RtcpPacket::parse(&bye).unwrap();

        assert_eq!(parsed.packet_type, RtcpType::Goodbye);
        assert_eq!(parsed.ssrc, 0xDEADBEEF);
    }

    #[test]
    fn test_parse_rtcp_too_short() {
        // Less than 8 bytes
        assert!(RtcpPacket::parse(&[0x80, 200, 0x00, 0x01]).is_none());
        assert!(RtcpPacket::parse(&[]).is_none());
    }

    #[test]
    fn test_parse_rtcp_invalid_version() {
        // Version 0
        let invalid = [0x00, 200, 0x00, 0x06, 0x12, 0x34, 0x56, 0x78];
        assert!(RtcpPacket::parse(&invalid).is_none());
    }

    #[test]
    fn test_parse_rtcp_unknown_type() {
        // Unknown packet type 199
        let unknown = [0x80, 199, 0x00, 0x01, 0x12, 0x34, 0x56, 0x78];
        assert!(RtcpPacket::parse(&unknown).is_none());
    }

    #[test]
    fn test_rtcp_type_from_byte() {
        assert_eq!(RtcpType::from_byte(200), Some(RtcpType::SenderReport));
        assert_eq!(RtcpType::from_byte(201), Some(RtcpType::ReceiverReport));
        assert_eq!(RtcpType::from_byte(202), Some(RtcpType::SourceDescription));
        assert_eq!(RtcpType::from_byte(203), Some(RtcpType::Goodbye));
        assert_eq!(RtcpType::from_byte(204), Some(RtcpType::ApplicationDefined));
        assert_eq!(RtcpType::from_byte(199), None);
        assert_eq!(RtcpType::from_byte(205), None);
    }

    #[test]
    fn test_rtcp_sr_structure() {
        let sr = build_rtcp_sr(0x12345678, 1000, 160000);

        // Length check: header(4) + SSRC(4) + NTP(8) + RTP-ts(4) + counts(8) = 28 bytes
        assert_eq!(sr.len(), 28);

        // Version and type
        assert_eq!(sr[0] >> 6, 2); // V=2
        assert_eq!(sr[1], 200); // PT=SR

        // Length field (in 32-bit words minus 1)
        let length = u16::from_be_bytes([sr[2], sr[3]]);
        assert_eq!(length, 6); // (28 - 4) / 4 = 6

        // Packet count
        let packets = u32::from_be_bytes([sr[20], sr[21], sr[22], sr[23]]);
        assert_eq!(packets, 1000);

        // Octet count
        let octets = u32::from_be_bytes([sr[24], sr[25], sr[26], sr[27]]);
        assert_eq!(octets, 160000);
    }

    #[test]
    fn test_rtcp_rr_structure() {
        let rr = build_rtcp_rr(0xAAAAAAAA, 0xBBBBBBBB, 64, 256, 0x00010064, 320);

        // Length check: header(4) + SSRC(4) + report_block(24) = 32 bytes
        assert_eq!(rr.len(), 32);

        // Loss fraction and cumulative
        let loss_word = u32::from_be_bytes([rr[12], rr[13], rr[14], rr[15]]);
        let loss_fraction = (loss_word >> 24) as u8;
        let cumulative_lost = loss_word & 0x00FFFFFF;
        assert_eq!(loss_fraction, 64);
        assert_eq!(cumulative_lost, 256);

        // Extended highest seq
        let ext_seq = u32::from_be_bytes([rr[16], rr[17], rr[18], rr[19]]);
        assert_eq!(ext_seq, 0x00010064);

        // Jitter
        let jitter = u32::from_be_bytes([rr[20], rr[21], rr[22], rr[23]]);
        assert_eq!(jitter, 320);
    }

    #[test]
    fn test_rtcp_bye_structure() {
        let bye = build_rtcp_bye(0xCAFEBABE);

        // SC=1 in first byte
        assert_eq!(bye[0] & 0x1F, 1);

        // PT=BYE (203)
        assert_eq!(bye[1], 203);

        // Length = 1
        let length = u16::from_be_bytes([bye[2], bye[3]]);
        assert_eq!(length, 1);
    }
}

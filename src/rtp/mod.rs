//! RTP (Real-time Transport Protocol) implementation.
//!
//! This module provides packet construction, parsing, and RTCP support
//! according to RFC 3550.
//!
//! # RTP Packet Structure
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |V=2|P|X|  CC   |M|     PT      |       sequence number         |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                           timestamp                           |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |           synchronization source (SSRC) identifier            |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```

mod header;
mod rtcp;
mod stats;

pub use header::{RtpHeader, RtpPacket};
pub use rtcp::{RtcpPacket, build_rtcp_bye, build_rtcp_rr, build_rtcp_sr};
pub use stats::{RtpCounters, RtpStats};

/// Parse an RTP packet header and return the payload type and payload data.
///
/// Returns `None` if the packet is too short or malformed.
pub fn parse_rtp(data: &[u8]) -> Option<(RtpHeader, &[u8])> {
    if data.len() < 12 {
        return None;
    }

    let header = RtpHeader::parse(data)?;
    let payload_offset = header.header_length();

    if payload_offset <= data.len() {
        Some((header, &data[payload_offset..]))
    } else {
        None
    }
}

/// Parse just the payload type from an RTP packet.
pub fn parse_payload_type(data: &[u8]) -> Option<u8> {
    if data.len() < 2 {
        return None;
    }
    Some(data[1] & 0x7F)
}

/// Parse the sequence number from an RTP packet.
pub fn parse_sequence(data: &[u8]) -> Option<u16> {
    if data.len() < 4 {
        return None;
    }
    Some(u16::from_be_bytes([data[2], data[3]]))
}

/// Parse the timestamp from an RTP packet.
pub fn parse_timestamp(data: &[u8]) -> Option<u32> {
    if data.len() < 8 {
        return None;
    }
    Some(u32::from_be_bytes([data[4], data[5], data[6], data[7]]))
}

/// Parse the SSRC from an RTP packet.
pub fn parse_ssrc(data: &[u8]) -> Option<u32> {
    if data.len() < 12 {
        return None;
    }
    Some(u32::from_be_bytes([data[8], data[9], data[10], data[11]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rtp() {
        let packet = [
            0x80, 0x00, // V=2, PT=0
            0x00, 0x01, // seq=1
            0x00, 0x00, 0x00, 0xA0, // timestamp=160
            0x12, 0x34, 0x56, 0x78, // SSRC
            0xDE, 0xAD, 0xBE, 0xEF, // payload
        ];

        let (header, payload) = parse_rtp(&packet).unwrap();
        assert_eq!(header.payload_type, 0);
        assert_eq!(header.sequence, 1);
        assert_eq!(header.timestamp, 160);
        assert_eq!(header.ssrc, 0x12345678);
        assert_eq!(payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_parse_helpers() {
        let packet = [
            0x80, 0x08, // V=2, PT=8
            0x00, 0x0A, // seq=10
            0x00, 0x00, 0x01, 0x00, // timestamp=256
            0xAA, 0xBB, 0xCC, 0xDD, // SSRC
        ];

        assert_eq!(parse_payload_type(&packet), Some(8));
        assert_eq!(parse_sequence(&packet), Some(10));
        assert_eq!(parse_timestamp(&packet), Some(256));
        assert_eq!(parse_ssrc(&packet), Some(0xAABBCCDD));
    }
}

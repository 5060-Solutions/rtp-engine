//! RTP header structures and parsing.

/// RTP packet header (RFC 3550).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpHeader {
    /// RTP version (always 2).
    pub version: u8,
    /// Padding flag.
    pub padding: bool,
    /// Extension flag.
    pub extension: bool,
    /// CSRC count.
    pub csrc_count: u8,
    /// Marker bit.
    pub marker: bool,
    /// Payload type (0-127).
    pub payload_type: u8,
    /// Sequence number (wraps at 65535).
    pub sequence: u16,
    /// Timestamp in clock rate units.
    pub timestamp: u32,
    /// Synchronization source identifier.
    pub ssrc: u32,
    /// Contributing source identifiers (0-15 entries).
    pub csrc: Vec<u32>,
    /// Header extension data (if extension flag is set).
    pub extension_data: Option<Vec<u8>>,
}

impl RtpHeader {
    /// Create a new RTP header with default values.
    pub fn new(payload_type: u8, sequence: u16, timestamp: u32, ssrc: u32) -> Self {
        Self {
            version: 2,
            padding: false,
            extension: false,
            csrc_count: 0,
            marker: false,
            payload_type,
            sequence,
            timestamp,
            ssrc,
            csrc: Vec::new(),
            extension_data: None,
        }
    }

    /// Create a header with the marker bit set (e.g., for first packet of a talkspurt).
    pub fn with_marker(mut self) -> Self {
        self.marker = true;
        self
    }

    /// Parse an RTP header from raw bytes.
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }

        let version = (data[0] >> 6) & 0x03;
        if version != 2 {
            return None; // Only RTP v2 is supported
        }

        let padding = (data[0] & 0x20) != 0;
        let extension = (data[0] & 0x10) != 0;
        let csrc_count = data[0] & 0x0F;
        let marker = (data[1] & 0x80) != 0;
        let payload_type = data[1] & 0x7F;
        let sequence = u16::from_be_bytes([data[2], data[3]]);
        let timestamp = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let ssrc = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);

        let mut header = Self {
            version,
            padding,
            extension,
            csrc_count,
            marker,
            payload_type,
            sequence,
            timestamp,
            ssrc,
            csrc: Vec::new(),
            extension_data: None,
        };

        // Parse CSRC list
        let csrc_end = 12 + (csrc_count as usize) * 4;
        if data.len() < csrc_end {
            return None;
        }
        for i in 0..csrc_count as usize {
            let offset = 12 + i * 4;
            let csrc = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            header.csrc.push(csrc);
        }

        // Parse extension header if present
        if extension {
            if data.len() < csrc_end + 4 {
                return None;
            }
            let ext_len = u16::from_be_bytes([data[csrc_end + 2], data[csrc_end + 3]]) as usize * 4;
            let ext_end = csrc_end + 4 + ext_len;
            if data.len() < ext_end {
                return None;
            }
            header.extension_data = Some(data[csrc_end..ext_end].to_vec());
        }

        Some(header)
    }

    /// Serialize the header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.csrc.len() * 4);

        // First byte: V=2, P, X, CC
        let byte0 = (self.version << 6)
            | (if self.padding { 0x20 } else { 0 })
            | (if self.extension { 0x10 } else { 0 })
            | (self.csrc.len() as u8 & 0x0F);
        buf.push(byte0);

        // Second byte: M, PT
        let byte1 = (if self.marker { 0x80 } else { 0 }) | (self.payload_type & 0x7F);
        buf.push(byte1);

        // Sequence number
        buf.extend_from_slice(&self.sequence.to_be_bytes());

        // Timestamp
        buf.extend_from_slice(&self.timestamp.to_be_bytes());

        // SSRC
        buf.extend_from_slice(&self.ssrc.to_be_bytes());

        // CSRC list
        for csrc in &self.csrc {
            buf.extend_from_slice(&csrc.to_be_bytes());
        }

        // Extension data
        if let Some(ref ext) = self.extension_data {
            buf.extend_from_slice(ext);
        }

        buf
    }

    /// Get the total header length in bytes.
    pub fn header_length(&self) -> usize {
        let base = 12 + self.csrc.len() * 4;
        if let Some(ref ext) = self.extension_data {
            base + ext.len()
        } else {
            base
        }
    }
}

impl Default for RtpHeader {
    fn default() -> Self {
        Self::new(0, 0, 0, 0)
    }
}

/// A complete RTP packet (header + payload).
#[derive(Debug, Clone)]
pub struct RtpPacket {
    /// The RTP header.
    pub header: RtpHeader,
    /// The payload data.
    pub payload: Vec<u8>,
}

impl RtpPacket {
    /// Create a new RTP packet.
    pub fn new(header: RtpHeader, payload: Vec<u8>) -> Self {
        Self { header, payload }
    }

    /// Create a packet with just the essential fields.
    pub fn simple(
        payload_type: u8,
        sequence: u16,
        timestamp: u32,
        ssrc: u32,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            header: RtpHeader::new(payload_type, sequence, timestamp, ssrc),
            payload,
        }
    }

    /// Parse an RTP packet from raw bytes.
    pub fn parse(data: &[u8]) -> Option<Self> {
        let header = RtpHeader::parse(data)?;
        let payload_offset = header.header_length();
        if payload_offset > data.len() {
            return None;
        }
        Some(Self {
            header,
            payload: data[payload_offset..].to_vec(),
        })
    }

    /// Serialize the packet to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = self.header.to_bytes();
        buf.extend_from_slice(&self.payload);
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = RtpHeader::new(8, 1234, 160000, 0xDEADBEEF);
        let bytes = header.to_bytes();
        let parsed = RtpHeader::parse(&bytes).unwrap();

        assert_eq!(header.payload_type, parsed.payload_type);
        assert_eq!(header.sequence, parsed.sequence);
        assert_eq!(header.timestamp, parsed.timestamp);
        assert_eq!(header.ssrc, parsed.ssrc);
    }

    #[test]
    fn test_header_with_marker() {
        let header = RtpHeader::new(0, 1, 160, 0x12345678).with_marker();
        let bytes = header.to_bytes();
        assert!(bytes[1] & 0x80 != 0);
    }

    #[test]
    fn test_packet_roundtrip() {
        let packet = RtpPacket::simple(0, 100, 16000, 0xABCDEF01, vec![1, 2, 3, 4, 5]);
        let bytes = packet.to_bytes();
        let parsed = RtpPacket::parse(&bytes).unwrap();

        assert_eq!(packet.header.sequence, parsed.header.sequence);
        assert_eq!(packet.payload, parsed.payload);
    }

    #[test]
    fn test_header_with_csrc() {
        let mut header = RtpHeader::new(0, 1, 160, 0x12345678);
        header.csrc.push(0xAAAAAAAA);
        header.csrc.push(0xBBBBBBBB);

        let bytes = header.to_bytes();
        let parsed = RtpHeader::parse(&bytes).unwrap();

        assert_eq!(parsed.csrc.len(), 2);
        assert_eq!(parsed.csrc[0], 0xAAAAAAAA);
        assert_eq!(parsed.csrc[1], 0xBBBBBBBB);
    }

    #[test]
    fn test_invalid_version() {
        // Version 0 packet
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xA0, 0x12, 0x34, 0x56, 0x78,
        ];
        assert!(RtpHeader::parse(&data).is_none());
    }

    #[test]
    fn test_packet_too_short() {
        // Less than 12 bytes
        assert!(RtpHeader::parse(&[0x80, 0x00, 0x00]).is_none());
        assert!(RtpHeader::parse(&[]).is_none());
    }

    #[test]
    fn test_header_default() {
        let header = RtpHeader::default();
        assert_eq!(header.version, 2);
        assert_eq!(header.payload_type, 0);
        assert_eq!(header.sequence, 0);
        assert_eq!(header.timestamp, 0);
        assert_eq!(header.ssrc, 0);
        assert!(!header.marker);
        assert!(!header.padding);
        assert!(!header.extension);
    }

    #[test]
    fn test_all_payload_types() {
        // Test all valid payload types (0-127)
        for pt in 0u8..128 {
            let header = RtpHeader::new(pt, 1, 160, 0x12345678);
            let bytes = header.to_bytes();
            let parsed = RtpHeader::parse(&bytes).unwrap();
            assert_eq!(parsed.payload_type, pt);
        }
    }

    #[test]
    fn test_sequence_wrap() {
        let header = RtpHeader::new(0, 65535, 160, 0x12345678);
        let bytes = header.to_bytes();
        let parsed = RtpHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.sequence, 65535);

        let header2 = RtpHeader::new(0, 0, 320, 0x12345678);
        let bytes2 = header2.to_bytes();
        let parsed2 = RtpHeader::parse(&bytes2).unwrap();
        assert_eq!(parsed2.sequence, 0);
    }

    #[test]
    fn test_timestamp_max() {
        let header = RtpHeader::new(0, 1, u32::MAX, 0x12345678);
        let bytes = header.to_bytes();
        let parsed = RtpHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.timestamp, u32::MAX);
    }

    #[test]
    fn test_header_length() {
        let mut header = RtpHeader::new(0, 1, 160, 0x12345678);
        assert_eq!(header.header_length(), 12);

        header.csrc.push(0xAAAAAAAA);
        assert_eq!(header.header_length(), 16);

        header.csrc.push(0xBBBBBBBB);
        assert_eq!(header.header_length(), 20);
    }

    #[test]
    fn test_padding_flag() {
        let mut header = RtpHeader::new(0, 1, 160, 0x12345678);
        header.padding = true;

        let bytes = header.to_bytes();
        assert!(bytes[0] & 0x20 != 0);

        let parsed = RtpHeader::parse(&bytes).unwrap();
        assert!(parsed.padding);
    }

    #[test]
    fn test_extension_flag() {
        let mut header = RtpHeader::new(0, 1, 160, 0x12345678);
        header.extension = true;
        // Extension data: 2-byte profile + 2-byte length (0) = 4 bytes
        header.extension_data = Some(vec![0xBE, 0xDE, 0x00, 0x00]);

        let bytes = header.to_bytes();
        assert!(bytes[0] & 0x10 != 0);

        let parsed = RtpHeader::parse(&bytes).unwrap();
        assert!(parsed.extension);
        assert!(parsed.extension_data.is_some());
    }

    #[test]
    fn test_csrc_count_max() {
        let mut header = RtpHeader::new(0, 1, 160, 0x12345678);
        // Add 15 CSRCs (max allowed by 4-bit CC field)
        for i in 0..15 {
            header.csrc.push(0x11111111 * (i + 1));
        }

        let bytes = header.to_bytes();
        let parsed = RtpHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.csrc.len(), 15);
    }

    #[test]
    fn test_truncated_csrc() {
        // Header claims 2 CSRCs but data is truncated
        let data = [
            0x82, 0x00, // V=2, CC=2
            0x00, 0x01, 0x00, 0x00, 0x00, 0xA0, 0x12, 0x34, 0x56, 0x78,
            // Only 4 bytes of CSRC instead of 8
            0xAA, 0xAA, 0xAA, 0xAA,
        ];
        assert!(RtpHeader::parse(&data).is_none());
    }

    #[test]
    fn test_packet_with_payload() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        let packet = RtpPacket::simple(0, 1, 160, 0x12345678, payload.clone());

        let bytes = packet.to_bytes();
        assert_eq!(bytes.len(), 12 + 8);

        let parsed = RtpPacket::parse(&bytes).unwrap();
        assert_eq!(parsed.payload, payload);
    }

    #[test]
    fn test_packet_empty_payload() {
        let packet = RtpPacket::simple(0, 1, 160, 0x12345678, vec![]);
        let bytes = packet.to_bytes();
        assert_eq!(bytes.len(), 12);

        let parsed = RtpPacket::parse(&bytes).unwrap();
        assert!(parsed.payload.is_empty());
    }

    #[test]
    fn test_rtp_packet_new() {
        let header = RtpHeader::new(8, 100, 16000, 0xABCDEF01);
        let packet = RtpPacket::new(header.clone(), vec![1, 2, 3]);

        assert_eq!(packet.header.payload_type, 8);
        assert_eq!(packet.payload, vec![1, 2, 3]);
    }
}

//! Audio codec implementations for RTP media.
//!
//! This module provides encoding and decoding for common VoIP codecs:
//!
//! - **G.711 μ-law (PCMU)**: PT 0, 8kHz, widely supported
//! - **G.711 A-law (PCMA)**: PT 8, 8kHz, common in Europe
//! - **Opus**: PT 111, 48kHz, modern high-quality codec (feature-gated)
//!
//! # Example
//!
//! ```
//! use rtp_engine::codec::{CodecType, create_encoder, create_decoder};
//!
//! let mut encoder = create_encoder(CodecType::Pcmu).unwrap();
//! let mut decoder = create_decoder(CodecType::Pcmu).unwrap();
//!
//! // Encode 160 samples (20ms at 8kHz)
//! let pcm: Vec<i16> = vec![0; 160];
//! let mut encoded = Vec::new();
//! encoder.encode(&pcm, &mut encoded);
//!
//! // Decode back to PCM
//! let mut decoded = Vec::new();
//! decoder.decode(&encoded, &mut decoded);
//! ```

mod g711;

#[cfg(feature = "opus")]
mod opus_codec;

pub use g711::{PcmaDecoder, PcmaEncoder, PcmuDecoder, PcmuEncoder};

#[cfg(feature = "opus")]
pub use opus_codec::{OpusDecoder, OpusEncoder};

use crate::error::Result;

/// Supported audio codec types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodecType {
    /// G.711 μ-law (PCMU) - RTP payload type 0
    Pcmu,
    /// G.711 A-law (PCMA) - RTP payload type 8
    Pcma,
    /// Opus - RTP payload type 111 (dynamic)
    #[cfg(feature = "opus")]
    Opus,
}

impl CodecType {
    /// Get the RTP payload type number for this codec.
    pub fn payload_type(&self) -> u8 {
        match self {
            Self::Pcmu => 0,
            Self::Pcma => 8,
            #[cfg(feature = "opus")]
            Self::Opus => 111,
        }
    }

    /// Get the codec name as used in SDP.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pcmu => "PCMU",
            Self::Pcma => "PCMA",
            #[cfg(feature = "opus")]
            Self::Opus => "opus",
        }
    }

    /// Get the clock rate (samples per second) for this codec.
    pub fn clock_rate(&self) -> u32 {
        match self {
            Self::Pcmu | Self::Pcma => 8000,
            #[cfg(feature = "opus")]
            Self::Opus => 48000,
        }
    }

    /// Get the number of channels for this codec.
    pub fn channels(&self) -> u8 {
        match self {
            Self::Pcmu | Self::Pcma => 1,
            #[cfg(feature = "opus")]
            Self::Opus => 1, // We use mono for VoIP
        }
    }

    /// Get the frame duration in milliseconds.
    pub fn frame_duration_ms(&self) -> u32 {
        20 // Standard 20ms frames for VoIP
    }

    /// Get the number of samples per frame.
    pub fn samples_per_frame(&self) -> usize {
        (self.clock_rate() * self.frame_duration_ms() / 1000) as usize
    }

    /// Parse a codec type from an RTP payload type number.
    pub fn from_payload_type(pt: u8) -> Option<Self> {
        match pt {
            0 => Some(Self::Pcmu),
            8 => Some(Self::Pcma),
            #[cfg(feature = "opus")]
            111 => Some(Self::Opus),
            _ => None,
        }
    }

    /// Parse a codec type from a codec name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "PCMU" | "G711U" => Some(Self::Pcmu),
            "PCMA" | "G711A" => Some(Self::Pcma),
            #[cfg(feature = "opus")]
            "OPUS" => Some(Self::Opus),
            _ => None,
        }
    }
}

impl std::fmt::Display for CodecType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for audio encoders.
///
/// Implementations encode PCM audio (signed 16-bit samples) to a compressed format.
pub trait AudioEncoder: Send {
    /// Encode PCM samples to the codec format.
    ///
    /// # Arguments
    /// * `pcm` - Input PCM samples (signed 16-bit, mono)
    /// * `output` - Output buffer for encoded data (appended to)
    ///
    /// # Returns
    /// Number of samples consumed from input.
    fn encode(&mut self, pcm: &[i16], output: &mut Vec<u8>) -> usize;

    /// Get the RTP payload type for this encoder.
    fn payload_type(&self) -> u8;

    /// Get the codec type.
    fn codec_type(&self) -> CodecType;
}

/// Trait for audio decoders.
///
/// Implementations decode compressed audio to PCM (signed 16-bit samples).
pub trait AudioDecoder: Send {
    /// Decode encoded data to PCM samples.
    ///
    /// # Arguments
    /// * `encoded` - Input encoded data
    /// * `output` - Output buffer for PCM samples (appended to)
    fn decode(&mut self, encoded: &[u8], output: &mut Vec<i16>);

    /// Get the codec type.
    fn codec_type(&self) -> CodecType;
}

/// Create an encoder for the specified codec type.
pub fn create_encoder(codec: CodecType) -> Result<Box<dyn AudioEncoder>> {
    match codec {
        CodecType::Pcmu => Ok(Box::new(PcmuEncoder::new())),
        CodecType::Pcma => Ok(Box::new(PcmaEncoder::new())),
        #[cfg(feature = "opus")]
        CodecType::Opus => Ok(Box::new(OpusEncoder::new()?)),
    }
}

/// Create a decoder for the specified codec type.
pub fn create_decoder(codec: CodecType) -> Result<Box<dyn AudioDecoder>> {
    match codec {
        CodecType::Pcmu => Ok(Box::new(PcmuDecoder::new())),
        CodecType::Pcma => Ok(Box::new(PcmaDecoder::new())),
        #[cfg(feature = "opus")]
        CodecType::Opus => Ok(Box::new(OpusDecoder::new()?)),
    }
}

/// Negotiate a codec from SDP answer.
///
/// Parses the `m=audio` line and returns the first mutually supported codec.
pub fn negotiate_codec(sdp: &str) -> CodecType {
    for line in sdp.lines() {
        let line = line.trim();
        if line.starts_with("m=audio ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // parts[3..] are payload types in priority order
            for pt_str in parts.iter().skip(3) {
                if let Ok(pt) = pt_str.parse::<u8>()
                    && let Some(codec) = CodecType::from_payload_type(pt)
                {
                    return codec;
                }
            }
        }
    }
    CodecType::Pcmu // fallback default
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_type_properties() {
        assert_eq!(CodecType::Pcmu.payload_type(), 0);
        assert_eq!(CodecType::Pcma.payload_type(), 8);
        assert_eq!(CodecType::Pcmu.clock_rate(), 8000);
        assert_eq!(CodecType::Pcmu.samples_per_frame(), 160);
    }

    #[test]
    fn test_codec_type_from_payload_type() {
        assert_eq!(CodecType::from_payload_type(0), Some(CodecType::Pcmu));
        assert_eq!(CodecType::from_payload_type(8), Some(CodecType::Pcma));
        assert_eq!(CodecType::from_payload_type(99), None);
    }

    #[test]
    fn test_codec_type_from_name() {
        assert_eq!(CodecType::from_name("PCMU"), Some(CodecType::Pcmu));
        assert_eq!(CodecType::from_name("pcma"), Some(CodecType::Pcma));
        assert_eq!(CodecType::from_name("unknown"), None);
    }

    #[test]
    fn test_negotiate_codec() {
        let sdp = "v=0\r\n\
                   m=audio 5004 RTP/AVP 0 8\r\n";
        assert_eq!(negotiate_codec(sdp), CodecType::Pcmu);

        let sdp2 = "v=0\r\n\
                    m=audio 5004 RTP/AVP 8 0\r\n";
        assert_eq!(negotiate_codec(sdp2), CodecType::Pcma);
    }

    #[test]
    fn test_negotiate_codec_fallback() {
        // No recognized codecs - should fall back to PCMU
        let sdp = "v=0\r\nm=audio 5004 RTP/AVP 96 97 98\r\n";
        assert_eq!(negotiate_codec(sdp), CodecType::Pcmu);
    }

    #[test]
    fn test_negotiate_codec_no_audio() {
        let sdp = "v=0\r\nm=video 5004 RTP/AVP 96\r\n";
        assert_eq!(negotiate_codec(sdp), CodecType::Pcmu); // fallback
    }

    #[test]
    fn test_codec_type_display() {
        assert_eq!(format!("{}", CodecType::Pcmu), "PCMU");
        assert_eq!(format!("{}", CodecType::Pcma), "PCMA");
    }

    #[test]
    fn test_codec_type_name() {
        assert_eq!(CodecType::Pcmu.name(), "PCMU");
        assert_eq!(CodecType::Pcma.name(), "PCMA");
    }

    #[test]
    fn test_codec_type_channels() {
        assert_eq!(CodecType::Pcmu.channels(), 1);
        assert_eq!(CodecType::Pcma.channels(), 1);
    }

    #[test]
    fn test_codec_type_frame_duration() {
        assert_eq!(CodecType::Pcmu.frame_duration_ms(), 20);
        assert_eq!(CodecType::Pcma.frame_duration_ms(), 20);
    }

    #[test]
    fn test_codec_from_name_aliases() {
        assert_eq!(CodecType::from_name("G711U"), Some(CodecType::Pcmu));
        assert_eq!(CodecType::from_name("G711A"), Some(CodecType::Pcma));
        assert_eq!(CodecType::from_name("g711u"), Some(CodecType::Pcmu));
    }

    #[test]
    fn test_create_encoder() {
        let encoder = create_encoder(CodecType::Pcmu).unwrap();
        assert_eq!(encoder.payload_type(), 0);
        assert_eq!(encoder.codec_type(), CodecType::Pcmu);

        let encoder = create_encoder(CodecType::Pcma).unwrap();
        assert_eq!(encoder.payload_type(), 8);
        assert_eq!(encoder.codec_type(), CodecType::Pcma);
    }

    #[test]
    fn test_create_decoder() {
        let decoder = create_decoder(CodecType::Pcmu).unwrap();
        assert_eq!(decoder.codec_type(), CodecType::Pcmu);

        let decoder = create_decoder(CodecType::Pcma).unwrap();
        assert_eq!(decoder.codec_type(), CodecType::Pcma);
    }

    #[test]
    fn test_encoder_empty_input() {
        let mut encoder = create_encoder(CodecType::Pcmu).unwrap();
        let mut output = Vec::new();
        let consumed = encoder.encode(&[], &mut output);
        assert_eq!(consumed, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_decoder_empty_input() {
        let mut decoder = create_decoder(CodecType::Pcmu).unwrap();
        let mut output = Vec::new();
        decoder.decode(&[], &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_codec_roundtrip_various_lengths() {
        let mut encoder = create_encoder(CodecType::Pcmu).unwrap();
        let mut decoder = create_decoder(CodecType::Pcmu).unwrap();

        for len in [1, 10, 80, 160, 320, 480] {
            let input: Vec<i16> = (0..len).map(|i| (i * 100) as i16).collect();
            let mut encoded = Vec::new();
            let consumed = encoder.encode(&input, &mut encoded);
            assert_eq!(consumed, len);
            assert_eq!(encoded.len(), len);

            let mut decoded = Vec::new();
            decoder.decode(&encoded, &mut decoded);
            assert_eq!(decoded.len(), len);
        }
    }

    #[test]
    fn test_codec_type_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CodecType::Pcmu);
        set.insert(CodecType::Pcma);
        set.insert(CodecType::Pcmu); // duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&CodecType::Pcmu));
        assert!(set.contains(&CodecType::Pcma));
    }

    #[test]
    fn test_codec_type_clone_copy() {
        let codec = CodecType::Pcmu;
        let copied = codec; // Copy trait
        let cloned = copied; // Also copy

        assert_eq!(codec, cloned);
        assert_eq!(codec, copied);
    }
}

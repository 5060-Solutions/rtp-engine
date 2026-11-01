//! Audio codec implementations for RTP media.
//!
//! This module provides encoding and decoding for common VoIP codecs:
//!
//! - **G.711 μ-law (PCMU)**: PT 0, 8kHz, 64kbps, universal support
//! - **G.711 A-law (PCMA)**: PT 8, 8kHz, 64kbps, common in Europe
//! - **G.729 framing**: PT 18, 8kHz, 8kbps. Not ITU-T G.729 compatible; see
//!   the `g729` module before negotiating it with any peer
//! - **G.722**: PT 9, 16kHz audio on an 8kHz RTP clock, 64kbps, wideband desk phones
//! - **Opus**: PT 111, 48kHz, modern high-quality codec (feature-gated)
//!
//! Every codec runs at its own native PCM rate: G.711 and G.729 take and
//! produce 8 kHz samples, Opus 48 kHz. [`CodecType::pcm_rate`] says which,
//! and [`CodecType::pcm_samples_per_packet`] how many samples make up one
//! 20 ms RTP packet at that rate. The RTP timestamp clock
//! ([`CodecType::clock_rate`]) is a separate thing: it happens to equal the
//! PCM rate for these codecs, but a codec like G.722 runs a 16 kHz PCM rate
//! on an 8 kHz RTP clock, so the two are kept apart.
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
mod g722;
mod g729;

#[cfg(feature = "opus")]
mod opus_codec;

pub use g711::{PcmaDecoder, PcmaEncoder, PcmuDecoder, PcmuEncoder};
pub use g722::{G722Decoder, G722Encoder};
pub use g729::{G729Decoder, G729Encoder};

#[cfg(feature = "opus")]
pub use opus_codec::{OpusDecoder, OpusEncoder};

use crate::error::Result;

/// Supported audio codec types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CodecType {
    /// G.711 μ-law (PCMU) - RTP payload type 0, 64kbps
    Pcmu,
    /// G.711 A-law (PCMA) - RTP payload type 8, 64kbps
    Pcma,
    /// G.729's framing and payload type 18, 8kbps. Not ITU-T G.729
    /// compatible: do not negotiate it with a peer that expects G.729.
    G729,
    /// G.722 - RTP payload type 9, 64kbps, 16 kHz audio on an 8 kHz RTP clock
    G722,
    /// Opus - RTP payload type 111 (dynamic), variable bitrate
    #[cfg(feature = "opus")]
    Opus,
}

impl CodecType {
    /// Get the RTP payload type number for this codec.
    pub fn payload_type(&self) -> u8 {
        match self {
            Self::Pcmu => 0,
            Self::Pcma => 8,
            Self::G729 => 18,
            Self::G722 => 9,
            #[cfg(feature = "opus")]
            Self::Opus => 111,
        }
    }

    /// Get the codec name as used in SDP.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pcmu => "PCMU",
            Self::Pcma => "PCMA",
            Self::G729 => "G729",
            Self::G722 => "G722",
            #[cfg(feature = "opus")]
            Self::Opus => "opus",
        }
    }

    /// The RTP timestamp clock rate for this codec, in ticks per second.
    ///
    /// This is the rate the `a=rtpmap` line names and the one RTP timestamps
    /// and jitter are measured in. For the sample rate of the PCM the encoder
    /// consumes and the decoder produces, see [`Self::pcm_rate`].
    pub fn clock_rate(&self) -> u32 {
        match self {
            // RFC 3551 §4.5.2: G.722's RTP clock is 8000 though its audio is 16 kHz.
            Self::Pcmu | Self::Pcma | Self::G729 | Self::G722 => 8000,
            #[cfg(feature = "opus")]
            Self::Opus => 48000,
        }
    }

    /// The sample rate of the PCM this codec encodes from and decodes to.
    ///
    /// G.711 and G.729 work on 8 kHz audio; Opus on 48 kHz. The media path
    /// resamples between the device rate and this rate, so Opus carries
    /// full-band audio instead of being squeezed through 8 kHz.
    pub fn pcm_rate(&self) -> u32 {
        match self {
            Self::Pcmu | Self::Pcma | Self::G729 => 8000,
            Self::G722 => 16000,
            #[cfg(feature = "opus")]
            Self::Opus => 48000,
        }
    }

    /// The packetization time every codec is sent with, in milliseconds.
    pub const PACKET_DURATION_MS: u32 = 20;

    /// PCM samples in one RTP packet at [`Self::pcm_rate`].
    ///
    /// Every codec is packetized at 20 ms: 160 samples for the 8 kHz codecs
    /// (two 10 ms G.729 frames), 960 for Opus. This is the exact length the
    /// encoder must be handed per packet — libopus rejects any other length
    /// and G.729 silently drops a partial frame.
    pub fn pcm_samples_per_packet(&self) -> usize {
        (self.pcm_rate() * Self::PACKET_DURATION_MS / 1000) as usize
    }

    /// How far the RTP timestamp advances per packet, in [`Self::clock_rate`]
    /// ticks: 160 for the 8 kHz codecs, 960 for Opus.
    pub fn rtp_ticks_per_packet(&self) -> u32 {
        self.clock_rate() * Self::PACKET_DURATION_MS / 1000
    }

    /// Get the number of channels for this codec.
    pub fn channels(&self) -> u8 {
        match self {
            Self::Pcmu | Self::Pcma | Self::G729 | Self::G722 => 1,
            #[cfg(feature = "opus")]
            Self::Opus => 1, // We use mono for VoIP
        }
    }

    /// The codec's own frame duration in milliseconds.
    ///
    /// This is the codec's internal unit (G.729 is 10 ms), not the packet
    /// size — see [`Self::pcm_samples_per_packet`] for that.
    pub fn frame_duration_ms(&self) -> u32 {
        match self {
            Self::G729 => 10, // G.729 uses 10ms frames
            _ => 20,          // Standard 20ms frames for VoIP
        }
    }

    /// PCM samples in one codec frame at [`Self::pcm_rate`].
    pub fn samples_per_frame(&self) -> usize {
        (self.pcm_rate() * self.frame_duration_ms() / 1000) as usize
    }

    /// Get the bitrate in kbps.
    pub fn bitrate_kbps(&self) -> u32 {
        match self {
            Self::Pcmu | Self::Pcma | Self::G722 => 64,
            Self::G729 => 8,
            #[cfg(feature = "opus")]
            Self::Opus => 32, // Typical for voice
        }
    }

    /// Parse a codec type from an RTP payload type number.
    pub fn from_payload_type(pt: u8) -> Option<Self> {
        match pt {
            0 => Some(Self::Pcmu),
            8 => Some(Self::Pcma),
            18 => Some(Self::G729),
            9 => Some(Self::G722),
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
            "G729" | "G.729" | "G729A" | "G.729A" | "G729B" | "G.729B" => Some(Self::G729),
            "G722" | "G.722" => Some(Self::G722),
            #[cfg(feature = "opus")]
            "OPUS" => Some(Self::Opus),
            _ => None,
        }
    }

    /// Get all available codec types.
    pub fn all() -> Vec<Self> {
        let mut codecs = vec![Self::Pcmu, Self::Pcma, Self::G729];
        codecs.push(Self::G722);
        #[cfg(feature = "opus")]
        codecs.push(Self::Opus);
        codecs
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

    /// Produce `samples` PCM samples standing in for a lost packet.
    ///
    /// `next_packet` is the packet after the lost one, when it has already
    /// arrived — a codec with in-band FEC (Opus) recovers the lost frame from
    /// it. The default appends silence, which is what the plain codecs do.
    fn conceal_loss(&mut self, samples: usize, next_packet: Option<&[u8]>, output: &mut Vec<i16>) {
        let _ = next_packet;
        output.extend(std::iter::repeat_n(0i16, samples));
    }

    /// Get the codec type.
    fn codec_type(&self) -> CodecType;
}

/// Create an encoder for the specified codec type.
pub fn create_encoder(codec: CodecType) -> Result<Box<dyn AudioEncoder>> {
    match codec {
        CodecType::Pcmu => Ok(Box::new(PcmuEncoder::new())),
        CodecType::Pcma => Ok(Box::new(PcmaEncoder::new())),
        CodecType::G729 => Ok(Box::new(G729Encoder::new())),
        CodecType::G722 => Ok(Box::new(G722Encoder::new())),
        #[cfg(feature = "opus")]
        CodecType::Opus => Ok(Box::new(OpusEncoder::new()?)),
    }
}

/// Create a decoder for the specified codec type.
pub fn create_decoder(codec: CodecType) -> Result<Box<dyn AudioDecoder>> {
    match codec {
        CodecType::Pcmu => Ok(Box::new(PcmuDecoder::new())),
        CodecType::Pcma => Ok(Box::new(PcmaDecoder::new())),
        CodecType::G729 => Ok(Box::new(G729Decoder::new())),
        CodecType::G722 => Ok(Box::new(G722Decoder::new())),
        #[cfg(feature = "opus")]
        CodecType::Opus => Ok(Box::new(OpusDecoder::new()?)),
    }
}

/// Negotiate a codec from SDP answer.
///
/// Parses the `m=audio` line and returns the first mutually supported codec.
/// Static payload types (0, 8, 18) match on their number; a dynamic payload
/// type matches on its `a=rtpmap` name and clock rate, so Opus is found on
/// any payload type a peer carries it on, not only 111. Falls back to PCMU
/// when nothing matched.
pub fn negotiate_codec(sdp: &str) -> CodecType {
    negotiate_codec_with_pt(sdp).map_or(CodecType::Pcmu, |(codec, _)| codec)
}

/// Negotiate a codec from SDP and report the payload type it is carried on.
///
/// Like [`negotiate_codec`], but returns `None` instead of a fallback when
/// nothing matched, and includes the payload type — which for a dynamic codec
/// is whatever number the peer chose, and is what the RTP packets must carry.
pub fn negotiate_codec_with_pt(sdp: &str) -> Option<(CodecType, u8)> {
    let mut offered: Vec<u8> = Vec::new();
    let mut rtpmaps: Vec<(u8, String, u32)> = Vec::new();

    for line in sdp.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("m=audio ") {
            // Only the first audio section counts; a second m=audio is a
            // separate stream this session does not carry.
            if offered.is_empty() {
                offered.extend(
                    rest.split_whitespace()
                        .skip(2)
                        .filter_map(|pt| pt.parse::<u8>().ok()),
                );
            }
        } else if let Some(rest) = line.strip_prefix("a=rtpmap:")
            && let Some((pt, codec)) = rest.split_once(char::is_whitespace)
            && let Ok(pt) = pt.trim().parse::<u8>()
        {
            let mut parts = codec.trim().split('/');
            let name = parts.next().unwrap_or_default().to_ascii_lowercase();
            if let Some(rate) = parts.next().and_then(|r| r.parse::<u32>().ok()) {
                rtpmaps.push((pt, name, rate));
            }
        }
    }

    for pt in offered {
        let rtpmap = rtpmaps.iter().find(|(map_pt, _, _)| *map_pt == pt);
        // An rtpmap line is authoritative for its payload type; without one,
        // the static assignments (and 111 as the conventional Opus number)
        // apply.
        let codec = match rtpmap {
            Some((_, name, rate)) => {
                CodecType::from_name(name).filter(|codec| codec.clock_rate() == *rate)
            }
            None => CodecType::from_payload_type(pt),
        };
        if let Some(codec) = codec {
            return Some((codec, pt));
        }
    }
    None
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
    fn pcm_rate_and_packet_size_follow_the_codec() {
        assert_eq!(CodecType::Pcmu.pcm_rate(), 8000);
        assert_eq!(CodecType::Pcma.pcm_rate(), 8000);
        assert_eq!(CodecType::G729.pcm_rate(), 8000);
        assert_eq!(CodecType::Pcmu.pcm_samples_per_packet(), 160);
        assert_eq!(CodecType::Pcma.pcm_samples_per_packet(), 160);
        // Two 10 ms G.729 frames make one 20 ms packet.
        assert_eq!(CodecType::G729.pcm_samples_per_packet(), 160);
        assert_eq!(CodecType::G729.samples_per_frame(), 80);
        #[cfg(feature = "opus")]
        {
            assert_eq!(CodecType::Opus.pcm_rate(), 48000);
            assert_eq!(CodecType::Opus.pcm_samples_per_packet(), 960);
        }
    }

    /// The RTP timestamp advances by one packet's worth of clock ticks, which
    /// for G.729 is 160 (20 ms at 8 kHz) and not its 80-sample frame — the
    /// stride that made its timestamp run at half speed.
    #[test]
    fn rtp_ticks_per_packet_is_twenty_ms_of_clock() {
        assert_eq!(CodecType::Pcmu.rtp_ticks_per_packet(), 160);
        assert_eq!(CodecType::G729.rtp_ticks_per_packet(), 160);
        #[cfg(feature = "opus")]
        assert_eq!(CodecType::Opus.rtp_ticks_per_packet(), 960);
    }

    /// The default concealment is silence of exactly the requested length.
    #[test]
    fn default_loss_concealment_is_silence() {
        let mut decoder = create_decoder(CodecType::Pcmu).unwrap();
        let mut out = vec![7i16];
        decoder.conceal_loss(160, None, &mut out);
        assert_eq!(out.len(), 161);
        assert!(out[1..].iter().all(|&s| s == 0));
    }

    #[cfg(feature = "opus")]
    #[test]
    fn negotiate_opus_by_rtpmap_name_on_a_dynamic_pt() {
        let sdp = "v=0\r\n\
                   m=audio 5004 RTP/AVP 96 0 8\r\n\
                   a=rtpmap:96 opus/48000/2\r\n\
                   a=rtpmap:0 PCMU/8000\r\n";
        assert_eq!(negotiate_codec(sdp), CodecType::Opus);
        assert_eq!(negotiate_codec_with_pt(sdp), Some((CodecType::Opus, 96)));
    }

    #[cfg(feature = "opus")]
    #[test]
    fn negotiate_opus_on_111_without_an_rtpmap() {
        let sdp = "m=audio 5004 RTP/AVP 111 0\r\n";
        assert_eq!(negotiate_codec_with_pt(sdp), Some((CodecType::Opus, 111)));
    }

    /// A dynamic payload type whose rtpmap names something else is not Opus,
    /// even on 111.
    #[test]
    fn negotiate_respects_an_rtpmap_that_reassigns_111() {
        let sdp = "m=audio 5004 RTP/AVP 111 8\r\n\
                   a=rtpmap:111 speex/16000\r\n\
                   a=rtpmap:8 PCMA/8000\r\n";
        assert_eq!(negotiate_codec_with_pt(sdp), Some((CodecType::Pcma, 8)));
    }

    #[test]
    fn negotiate_static_pts_still_match_without_rtpmap() {
        let sdp = "m=audio 5004 RTP/AVP 18 0\r\n";
        assert_eq!(negotiate_codec_with_pt(sdp), Some((CodecType::G729, 18)));
    }

    #[test]
    fn negotiate_with_pt_reports_none_when_nothing_matches() {
        let sdp = "m=audio 5004 RTP/AVP 96\r\na=rtpmap:96 speex/16000\r\n";
        assert_eq!(negotiate_codec_with_pt(sdp), None);
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
        assert_eq!(output, [] as [u8; 0]);
    }

    #[test]
    fn test_decoder_empty_input() {
        let mut decoder = create_decoder(CodecType::Pcmu).unwrap();
        let mut output = Vec::new();
        decoder.decode(&[], &mut output);
        assert_eq!(output, [] as [i16; 0]);
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

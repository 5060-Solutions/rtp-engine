//! Opus codec implementation.
//!
//! Opus is a modern, high-quality codec optimized for voice and music.
//! It operates at 48kHz internally, so resampling is required for 8kHz VoIP.

use super::{AudioDecoder, AudioEncoder, CodecType};
use crate::error::{Error, Result};

/// Opus encoder with integrated 8kHz→48kHz resampling.
pub struct OpusEncoder {
    encoder: audiopus::coder::Encoder,
    resample_buf: Vec<i16>,
}

impl OpusEncoder {
    /// Create a new Opus encoder.
    pub fn new() -> Result<Self> {
        let encoder = audiopus::coder::Encoder::new(
            audiopus::SampleRate::Hz48000,
            audiopus::Channels::Mono,
            audiopus::Application::Voip,
        )
        .map_err(|e| Error::codec(format!("Opus encoder init: {}", e)))?;

        Ok(Self {
            encoder,
            resample_buf: Vec::with_capacity(960),
        })
    }
}

impl AudioEncoder for OpusEncoder {
    fn encode(&mut self, pcm: &[i16], output: &mut Vec<u8>) -> usize {
        // Resample 8kHz → 48kHz (6x linear interpolation)
        self.resample_buf.clear();
        for i in 0..pcm.len() {
            let current = pcm[i] as f64;
            let next = if i + 1 < pcm.len() {
                pcm[i + 1] as f64
            } else {
                current
            };
            for j in 0..6 {
                let t = j as f64 / 6.0;
                self.resample_buf
                    .push((current + (next - current) * t) as i16);
            }
        }

        let mut encoded = [0u8; 4000];
        match self.encoder.encode(&self.resample_buf, &mut encoded) {
            Ok(len) => {
                output.extend_from_slice(&encoded[..len]);
                pcm.len()
            }
            Err(e) => {
                log::error!("Opus encode error: {}", e);
                0
            }
        }
    }

    fn payload_type(&self) -> u8 {
        111
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Opus
    }
}

impl std::fmt::Debug for OpusEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusEncoder")
            .field("resample_buf_capacity", &self.resample_buf.capacity())
            .finish()
    }
}

/// Opus decoder with integrated 48kHz→8kHz resampling.
pub struct OpusDecoder {
    decoder: audiopus::coder::Decoder,
}

impl OpusDecoder {
    /// Create a new Opus decoder.
    pub fn new() -> Result<Self> {
        let decoder =
            audiopus::coder::Decoder::new(audiopus::SampleRate::Hz48000, audiopus::Channels::Mono)
                .map_err(|e| Error::codec(format!("Opus decoder init: {}", e)))?;

        Ok(Self { decoder })
    }
}

impl AudioDecoder for OpusDecoder {
    fn decode(&mut self, encoded: &[u8], output: &mut Vec<i16>) {
        let mut decoded = [0i16; 5760]; // Max opus frame

        let packet: audiopus::packet::Packet<'_> = match encoded.try_into() {
            Ok(p) => p,
            Err(e) => {
                log::error!("Opus packet error: {}", e);
                return;
            }
        };
        let out_signals: audiopus::MutSignals<'_, _> = match (&mut decoded[..]).try_into() {
            Ok(s) => s,
            Err(e) => {
                log::error!("Opus signals error: {}", e);
                return;
            }
        };

        match self.decoder.decode(Some(packet), out_signals, false) {
            Ok(samples) => {
                // Downsample 48kHz → 8kHz (take every 6th sample)
                for i in (0..samples).step_by(6) {
                    output.push(decoded[i]);
                }
            }
            Err(e) => {
                log::error!("Opus decode error: {}", e);
            }
        }
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Opus
    }
}

impl std::fmt::Debug for OpusDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusDecoder").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_encoder_creation() {
        let encoder = OpusEncoder::new();
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_decoder_creation() {
        let decoder = OpusDecoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_opus_roundtrip() {
        let mut encoder = OpusEncoder::new().unwrap();
        let mut decoder = OpusDecoder::new().unwrap();

        // Create a 20ms frame of 440Hz sine wave at 8kHz
        let input: Vec<i16> = (0..160)
            .map(|i| {
                let t = i as f64 / 8000.0;
                (f64::sin(2.0 * std::f64::consts::PI * 440.0 * t) * 16000.0) as i16
            })
            .collect();

        let mut encoded = Vec::new();
        let consumed = encoder.encode(&input, &mut encoded);
        assert_eq!(consumed, 160);
        assert!(!encoded.is_empty());

        let mut decoded = Vec::new();
        decoder.decode(&encoded, &mut decoded);
        // Opus output should be roughly the same length after resampling
        assert!(decoded.len() >= 100 && decoded.len() <= 200);
    }
}

//! G.711 codec implementations (μ-law and A-law).
//!
//! G.711 is the most widely supported VoIP codec, providing 64 kbps audio
//! at 8kHz sample rate with 8-bit samples.

use super::{AudioDecoder, AudioEncoder, CodecType};

/// G.711 μ-law (PCMU) encoder.
///
/// Compresses 16-bit PCM to 8-bit μ-law samples (1:2 compression).
#[derive(Debug, Default)]
pub struct PcmuEncoder;

impl PcmuEncoder {
    /// Create a new PCMU encoder.
    pub fn new() -> Self {
        Self
    }
}

impl AudioEncoder for PcmuEncoder {
    fn encode(&mut self, pcm: &[i16], output: &mut Vec<u8>) -> usize {
        output.reserve(pcm.len());
        for &sample in pcm {
            output.push(linear_to_ulaw(sample));
        }
        pcm.len()
    }

    fn payload_type(&self) -> u8 {
        0
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Pcmu
    }
}

/// G.711 μ-law (PCMU) decoder.
///
/// Expands 8-bit μ-law samples to 16-bit PCM.
#[derive(Debug, Default)]
pub struct PcmuDecoder;

impl PcmuDecoder {
    /// Create a new PCMU decoder.
    pub fn new() -> Self {
        Self
    }
}

impl AudioDecoder for PcmuDecoder {
    fn decode(&mut self, encoded: &[u8], output: &mut Vec<i16>) {
        output.reserve(encoded.len());
        for &b in encoded {
            output.push(ulaw_to_linear(b));
        }
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Pcmu
    }
}

/// G.711 A-law (PCMA) encoder.
///
/// Compresses 16-bit PCM to 8-bit A-law samples (1:2 compression).
#[derive(Debug, Default)]
pub struct PcmaEncoder;

impl PcmaEncoder {
    /// Create a new PCMA encoder.
    pub fn new() -> Self {
        Self
    }
}

impl AudioEncoder for PcmaEncoder {
    fn encode(&mut self, pcm: &[i16], output: &mut Vec<u8>) -> usize {
        output.reserve(pcm.len());
        for &sample in pcm {
            output.push(linear_to_alaw(sample));
        }
        pcm.len()
    }

    fn payload_type(&self) -> u8 {
        8
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Pcma
    }
}

/// G.711 A-law (PCMA) decoder.
///
/// Expands 8-bit A-law samples to 16-bit PCM.
#[derive(Debug, Default)]
pub struct PcmaDecoder;

impl PcmaDecoder {
    /// Create a new PCMA decoder.
    pub fn new() -> Self {
        Self
    }
}

impl AudioDecoder for PcmaDecoder {
    fn decode(&mut self, encoded: &[u8], output: &mut Vec<i16>) {
        output.reserve(encoded.len());
        for &b in encoded {
            output.push(alaw_to_linear(b));
        }
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Pcma
    }
}

// --- G.711 μ-law implementation (ITU-T G.711) ---

const ULAW_BIAS: i16 = 0x84;
const ULAW_MAX: i16 = 0x7FFF;

/// Convert a 16-bit linear PCM sample to 8-bit μ-law.
fn linear_to_ulaw(sample: i16) -> u8 {
    let sign: u8;
    let mut pcm = sample;

    if pcm < 0 {
        sign = 0x80;
        pcm = if pcm == i16::MIN { ULAW_MAX } else { -pcm };
    } else {
        sign = 0;
    }

    pcm = pcm.saturating_add(ULAW_BIAS);

    let mut exponent = 7u8;
    let mut mask = 0x4000i16;
    while exponent > 0 && (pcm & mask) == 0 {
        exponent -= 1;
        mask >>= 1;
    }

    let mantissa = ((pcm >> (exponent + 3)) & 0x0F) as u8;
    !(sign | (exponent << 4) | mantissa)
}

/// Convert an 8-bit μ-law sample to 16-bit linear PCM.
fn ulaw_to_linear(ulaw: u8) -> i16 {
    let ulaw = !ulaw;
    let sign = ulaw & 0x80;
    let exponent = ((ulaw >> 4) & 0x07) as i32;
    let mantissa = (ulaw & 0x0F) as i32;
    let mut sample = ((mantissa << 3) + ULAW_BIAS as i32) << exponent;
    sample -= ULAW_BIAS as i32;
    if sign != 0 {
        -sample as i16
    } else {
        sample as i16
    }
}

// --- G.711 A-law implementation (ITU-T G.711) ---

const ALAW_SEG_END: [i16; 8] = [0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF];

/// Convert a 16-bit linear PCM sample to 8-bit A-law.
fn linear_to_alaw(sample: i16) -> u8 {
    let sign: u8;
    let abs_sample: i16;

    if sample >= 0 {
        sign = 0xD5;
        abs_sample = sample;
    } else {
        sign = 0x55;
        abs_sample = if sample == i16::MIN {
            i16::MAX
        } else {
            -sample
        };
    }

    let sample = abs_sample >> 3;

    let mut seg = 0usize;
    while seg < 8 && sample > ALAW_SEG_END[seg] {
        seg += 1;
    }

    let aval = if seg >= 8 {
        0x7F
    } else if seg < 2 {
        (sample >> 1) as u8
    } else {
        ((seg as u8) << 4) | ((sample >> seg) & 0x0F) as u8
    };

    aval ^ sign
}

/// Convert an 8-bit A-law sample to 16-bit linear PCM.
fn alaw_to_linear(alaw: u8) -> i16 {
    let alaw = alaw ^ 0x55;
    let seg = ((alaw & 0x70) >> 4) as i32;
    let value = ((alaw & 0x0F) << 1) | 1;

    let sample = if seg == 0 {
        (value as i32) << 3
    } else {
        ((value as i32) | 0x20) << (seg + 2)
    };

    if (alaw & 0x80) != 0 {
        sample as i16
    } else {
        -(sample as i16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ulaw_roundtrip() {
        // Test roundtrip for a range of values
        for sample in [
            -32768i16, -16384, -8192, -1024, -128, 0, 128, 1024, 8192, 16384, 32767,
        ] {
            let encoded = linear_to_ulaw(sample);
            let decoded = ulaw_to_linear(encoded);
            // G.711 is lossy, but should be reasonably close
            let diff = (sample as i32 - decoded as i32).abs();
            assert!(
                diff < 2048,
                "μ-law roundtrip too lossy: {} -> {} (diff {})",
                sample,
                decoded,
                diff
            );
        }
    }

    #[test]
    fn test_alaw_roundtrip() {
        for sample in [
            -32768i16, -16384, -8192, -1024, -128, 0, 128, 1024, 8192, 16384, 32767,
        ] {
            let encoded = linear_to_alaw(sample);
            let decoded = alaw_to_linear(encoded);
            let diff = (sample as i32 - decoded as i32).abs();
            assert!(
                diff < 2048,
                "A-law roundtrip too lossy: {} -> {} (diff {})",
                sample,
                decoded,
                diff
            );
        }
    }

    #[test]
    fn test_ulaw_encoder_decoder() {
        let mut encoder = PcmuEncoder::new();
        let mut decoder = PcmuDecoder::new();

        let input: Vec<i16> = (0..160)
            .map(|i| ((i as f32 / 160.0) * 16000.0) as i16)
            .collect();
        let mut encoded = Vec::new();
        let consumed = encoder.encode(&input, &mut encoded);

        assert_eq!(consumed, 160);
        assert_eq!(encoded.len(), 160);

        let mut decoded = Vec::new();
        decoder.decode(&encoded, &mut decoded);
        assert_eq!(decoded.len(), 160);
    }

    #[test]
    fn test_alaw_encoder_decoder() {
        let mut encoder = PcmaEncoder::new();
        let mut decoder = PcmaDecoder::new();

        let input: Vec<i16> = (0..160)
            .map(|i| ((i as f32 / 160.0) * 16000.0) as i16)
            .collect();
        let mut encoded = Vec::new();
        let consumed = encoder.encode(&input, &mut encoded);

        assert_eq!(consumed, 160);
        assert_eq!(encoded.len(), 160);

        let mut decoded = Vec::new();
        decoder.decode(&encoded, &mut decoded);
        assert_eq!(decoded.len(), 160);
    }

    #[test]
    fn test_silence_encoding() {
        let mut encoder = PcmuEncoder::new();
        let silence: Vec<i16> = vec![0; 160];
        let mut encoded = Vec::new();
        encoder.encode(&silence, &mut encoded);

        // μ-law silence is 0xFF (127 biased and inverted)
        assert!(encoded.iter().all(|&b| b == 0xFF || b == 0x7F));
    }

    #[test]
    fn test_ulaw_extreme_values() {
        // Test the most extreme values
        let max_encoded = linear_to_ulaw(i16::MAX);
        let min_encoded = linear_to_ulaw(i16::MIN);

        // Should encode to different values
        assert_ne!(max_encoded, min_encoded);

        // Decode back should preserve polarity
        let max_decoded = ulaw_to_linear(max_encoded);
        let min_decoded = ulaw_to_linear(min_encoded);

        assert!(max_decoded > 0);
        assert!(min_decoded < 0);
    }

    #[test]
    fn test_alaw_extreme_values() {
        let max_encoded = linear_to_alaw(i16::MAX);
        let min_encoded = linear_to_alaw(i16::MIN);

        assert_ne!(max_encoded, min_encoded);

        let max_decoded = alaw_to_linear(max_encoded);
        let min_decoded = alaw_to_linear(min_encoded);

        assert!(max_decoded > 0);
        assert!(min_decoded < 0);
    }

    #[test]
    fn test_ulaw_monotonicity() {
        // μ-law should be monotonic for positive values
        let mut last_encoded = linear_to_ulaw(0);
        for sample in (0..32768i32).step_by(256) {
            let encoded = linear_to_ulaw(sample as i16);
            // Encoded values for increasing input should not increase
            // (due to complementing in μ-law encoding)
            assert!(
                encoded <= last_encoded || sample < 256,
                "μ-law not monotonic at {}: {} > {}",
                sample,
                encoded,
                last_encoded
            );
            last_encoded = encoded;
        }
    }

    #[test]
    fn test_pcmu_encoder_properties() {
        let encoder = PcmuEncoder::new();

        assert_eq!(encoder.codec_type(), CodecType::Pcmu);
        assert_eq!(encoder.payload_type(), 0);
    }

    #[test]
    fn test_pcma_encoder_properties() {
        let encoder = PcmaEncoder::new();

        assert_eq!(encoder.codec_type(), CodecType::Pcma);
        assert_eq!(encoder.payload_type(), 8);
    }

    #[test]
    fn test_pcmu_decoder_properties() {
        let decoder = PcmuDecoder::new();

        assert_eq!(decoder.codec_type(), CodecType::Pcmu);
    }

    #[test]
    fn test_pcma_decoder_properties() {
        let decoder = PcmaDecoder::new();

        assert_eq!(decoder.codec_type(), CodecType::Pcma);
    }

    #[test]
    fn test_encoder_partial_frame() {
        let mut encoder = PcmuEncoder::new();

        // Less than one frame
        let input: Vec<i16> = vec![1000; 80];
        let mut encoded = Vec::new();
        let consumed = encoder.encode(&input, &mut encoded);

        assert_eq!(consumed, 80);
        assert_eq!(encoded.len(), 80);
    }

    #[test]
    fn test_encoder_multiple_frames() {
        let mut encoder = PcmuEncoder::new();

        // Multiple frames
        let input: Vec<i16> = vec![1000; 480];
        let mut encoded = Vec::new();
        let consumed = encoder.encode(&input, &mut encoded);

        assert_eq!(consumed, 480);
        assert_eq!(encoded.len(), 480);
    }

    #[test]
    fn test_ulaw_all_codewords() {
        // Test that all 256 codewords decode without panic
        for codeword in 0u8..=255 {
            let _decoded = ulaw_to_linear(codeword);
            // If we got here without panic, the codeword is valid
        }
    }

    #[test]
    fn test_alaw_all_codewords() {
        // Test that all 256 codewords decode without panic
        for codeword in 0u8..=255 {
            let _decoded = alaw_to_linear(codeword);
            // If we got here without panic, the codeword is valid
        }
    }

    #[test]
    fn test_sine_wave_encoding() {
        let mut encoder = PcmuEncoder::new();
        let mut decoder = PcmuDecoder::new();

        // Generate a 1kHz sine wave at 8kHz sample rate
        let samples: Vec<i16> = (0..160)
            .map(|i| {
                let t = i as f32 / 8000.0;
                (1000.0 * (2.0 * std::f32::consts::PI * t).sin() * 16000.0) as i16
            })
            .collect();

        let mut encoded = Vec::new();
        encoder.encode(&samples, &mut encoded);

        let mut decoded = Vec::new();
        decoder.decode(&encoded, &mut decoded);

        // Check that decoded signal has similar characteristics
        assert_eq!(decoded.len(), samples.len());

        // Calculate correlation between original and decoded
        let correlation: f64 = samples
            .iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        // Should have positive correlation (same polarity/phase)
        assert!(
            correlation > 0.0,
            "Decoded signal not correlated with original"
        );
    }
}

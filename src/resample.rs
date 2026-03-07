//! Audio resampling utilities.
//!
//! Provides sample rate conversion between codec rates (typically 8kHz)
//! and device rates (typically 44.1kHz or 48kHz).
//!
//! For streaming audio (RTP), use `StreamResampler` which maintains state across
//! calls for glitch-free frame-boundary transitions.
//!
//! For one-shot resampling, use `resample_linear`.

use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Async, FixedAsync, PolynomialDegree, Resampler};

/// Resample audio from one sample rate to another using high-quality FFT interpolation.
///
/// Uses FFT-based synchronous resampling for high-quality results.
/// Falls back to linear interpolation for very small chunks or on error.
///
/// # Arguments
/// * `input` - Input samples
/// * `from_rate` - Source sample rate in Hz
/// * `to_rate` - Target sample rate in Hz
///
/// # Returns
/// Resampled audio samples.
pub fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }

    // For very small chunks, fall back to simple linear interpolation
    if input.len() < 64 {
        return resample_linear_simple(input, from_rate, to_rate);
    }

    // Try to use rubato for high-quality resampling
    match resample_with_rubato(input, from_rate, to_rate) {
        Ok(output) => output,
        Err(e) => {
            log::warn!("Rubato resampling failed ({}), falling back to linear", e);
            resample_linear_simple(input, from_rate, to_rate)
        }
    }
}

/// High-quality resampling using rubato asynchronous resampler.
/// Uses cubic polynomial interpolation for good quality with low latency.
fn resample_with_rubato(input: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, String> {
    let ratio = to_rate as f64 / from_rate as f64;

    // Create an asynchronous resampler with polynomial interpolation (no sinc for speed)
    // Using FixedAsync::Input so our input chunk size is fixed
    let mut resampler = Async::<f32>::new_poly(
        ratio,                   // resample ratio (output/input)
        1.0,                     // max_resample_ratio_relative (no dynamic adjustment)
        PolynomialDegree::Cubic, // cubic interpolation for good quality
        input.len(),             // chunk size = input length
        1,                       // mono
        FixedAsync::Input,       // fixed input size
    )
    .map_err(|e| format!("Failed to create resampler: {:?}", e))?;

    // Calculate expected output length
    let expected_output_len = ((input.len() as f64) * ratio).ceil() as usize;

    // Prepare input as single-channel vector of vectors
    let input_vec = vec![input.to_vec()];
    let input_adapter = SequentialSliceOfVecs::new(&input_vec, 1, input.len())
        .map_err(|e| format!("Input adapter error: {}", e))?;

    // Prepare output buffer with headroom
    let buffer_len = expected_output_len * 2 + 1024;
    let mut output_vec = vec![vec![0.0f32; buffer_len]];
    let mut output_adapter = SequentialSliceOfVecs::new_mut(&mut output_vec, 1, buffer_len)
        .map_err(|e| format!("Output adapter error: {}", e))?;

    // Process all samples at once
    let (_, frames_written) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, input.len(), None)
        .map_err(|e| format!("Resample error: {:?}", e))?;

    // Extract output and truncate to actual written frames
    let mut result = output_vec.into_iter().next().unwrap_or_default();
    result.truncate(frames_written);

    Ok(result)
}

/// Simple linear interpolation fallback for small chunks or errors.
fn resample_linear_simple(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx_f = i as f64 * ratio;
        let src_idx = src_idx_f as usize;
        let frac = src_idx_f - src_idx as f64;

        let s0 = input.get(src_idx).copied().unwrap_or(0.0) as f64;
        let s1 = input
            .get(src_idx + 1)
            .copied()
            .unwrap_or_else(|| input.get(src_idx).copied().unwrap_or(0.0)) as f64;

        let sample = s0 + frac * (s1 - s0);
        output.push(sample as f32);
    }

    output
}

/// Resample i16 PCM audio from one sample rate to another.
/// Uses high-quality resampling via f32 conversion.
pub fn resample_linear_i16(input: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }

    // Convert to f32, resample, convert back
    let f32_input = i16_to_f32(input);
    let f32_output = resample_linear(&f32_input, from_rate, to_rate);
    f32_to_i16(&f32_output)
}

/// Convert f32 samples (-1.0 to 1.0) to i16 samples with proper clamping.
pub fn f32_to_i16(input: &[f32]) -> Vec<i16> {
    input
        .iter()
        .map(|&s| {
            // Clamp to [-1.0, 1.0] range before conversion
            let clamped = s.clamp(-1.0, 1.0);
            (clamped * 32767.0) as i16
        })
        .collect()
}

/// Convert i16 samples to f32 samples (-1.0 to 1.0).
pub fn i16_to_f32(input: &[i16]) -> Vec<f32> {
    input.iter().map(|&s| s as f32 / 32768.0).collect()
}

/// A streaming resampler that maintains state across calls for glitch-free audio.
///
/// Unlike `resample_linear` which creates a fresh resampler per call (causing
/// discontinuities at frame boundaries), this struct keeps the rubato resampler
/// alive so interpolation state carries over between frames.
///
/// Use this for continuous audio streams (RTP TX/RX paths).
pub struct StreamResampler {
    resampler: Async<f32>,
    from_rate: u32,
    to_rate: u32,
    chunk_size: usize,
    /// Leftover input samples from previous call (when input isn't exact chunk_size)
    remainder: Vec<f32>,
}

impl StreamResampler {
    /// Create a new streaming resampler.
    ///
    /// * `from_rate` - Source sample rate (e.g. 8000 for codec, 48000 for device)
    /// * `to_rate` - Target sample rate
    /// * `chunk_size` - Expected input chunk size per call (e.g. 160 for 20ms at 8kHz)
    pub fn new(from_rate: u32, to_rate: u32, chunk_size: usize) -> Self {
        let ratio = to_rate as f64 / from_rate as f64;
        let resampler = Async::<f32>::new_poly(
            ratio,
            1.0,
            PolynomialDegree::Cubic,
            chunk_size,
            1,
            FixedAsync::Input,
        )
        .expect("Failed to create streaming resampler");

        Self {
            resampler,
            from_rate,
            to_rate,
            chunk_size,
            remainder: Vec::new(),
        }
    }

    /// Process a chunk of audio samples, returning resampled output.
    ///
    /// Handles variable-length input by buffering remainders internally.
    /// State is preserved between calls for continuous interpolation.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.from_rate == self.to_rate || input.is_empty() {
            return input.to_vec();
        }

        // Append to remainder from previous call
        self.remainder.extend_from_slice(input);

        let mut output = Vec::new();

        // Process full chunks
        while self.remainder.len() >= self.chunk_size {
            let chunk: Vec<f32> = self.remainder.drain(..self.chunk_size).collect();
            match self.process_chunk(&chunk) {
                Ok(resampled) => output.extend(resampled),
                Err(_) => {
                    // Fallback to simple linear for this chunk
                    output.extend(resample_linear_simple(&chunk, self.from_rate, self.to_rate));
                }
            }
        }

        output
    }

    /// Flush any remaining samples (call when stream ends).
    pub fn flush(&mut self) -> Vec<f32> {
        if self.remainder.is_empty() {
            return Vec::new();
        }
        let remaining = std::mem::take(&mut self.remainder);
        resample_linear_simple(&remaining, self.from_rate, self.to_rate)
    }

    fn process_chunk(&mut self, chunk: &[f32]) -> Result<Vec<f32>, String> {
        let ratio = self.to_rate as f64 / self.from_rate as f64;
        let expected_output_len = ((chunk.len() as f64) * ratio).ceil() as usize;
        let buffer_len = expected_output_len + 64;

        let input_vec = vec![chunk.to_vec()];
        let input_adapter = SequentialSliceOfVecs::new(&input_vec, 1, chunk.len())
            .map_err(|e| format!("Input adapter error: {}", e))?;

        let mut output_vec = vec![vec![0.0f32; buffer_len]];
        let mut output_adapter = SequentialSliceOfVecs::new_mut(&mut output_vec, 1, buffer_len)
            .map_err(|e| format!("Output adapter error: {}", e))?;

        let (_, frames_written) = self
            .resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, None)
            .map_err(|e| format!("Resample error: {:?}", e))?;

        let mut result = output_vec.into_iter().next().unwrap_or_default();
        result.truncate(frames_written);
        Ok(result)
    }
}

impl std::fmt::Debug for StreamResampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamResampler")
            .field("from_rate", &self.from_rate)
            .field("to_rate", &self.to_rate)
            .field("chunk_size", &self.chunk_size)
            .field("remainder", &self.remainder.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_upsample() {
        // Upsample 8kHz to 48kHz (6x)
        let input: Vec<f32> = (0..160).map(|i| (i as f32 / 160.0) * 2.0 - 1.0).collect();
        let output = resample_linear(&input, 8000, 48000);

        // Should have roughly 6x more samples
        assert!(
            output.len() >= 800,
            "Expected at least 800, got {}",
            output.len()
        );
        assert!(
            output.len() <= 1100,
            "Expected at most 1100, got {}",
            output.len()
        );
    }

    #[test]
    fn test_resample_downsample() {
        // Downsample 48kHz to 8kHz (1/6x)
        let input: Vec<f32> = (0..960).map(|i| (i as f32 / 960.0) * 2.0 - 1.0).collect();
        let output = resample_linear(&input, 48000, 8000);

        // Should have roughly 1/6 the samples
        assert!(
            output.len() >= 140,
            "Expected at least 140, got {}",
            output.len()
        );
        assert!(
            output.len() <= 180,
            "Expected at most 180, got {}",
            output.len()
        );
    }

    #[test]
    fn test_resample_same_rate() {
        let input: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let output = resample_linear(&input, 8000, 8000);

        assert_eq!(input, output);
    }

    #[test]
    fn test_resample_empty() {
        let input: Vec<f32> = vec![];
        let output = resample_linear(&input, 8000, 48000);

        assert!(output.is_empty());
    }

    #[test]
    fn test_f32_i16_conversion() {
        let f32_samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let i16_samples = f32_to_i16(&f32_samples);

        assert_eq!(i16_samples[0], 0);
        assert!((i16_samples[1] - 16383).abs() < 2);
        assert!((i16_samples[2] + 16383).abs() < 2);

        let back = i16_to_f32(&i16_samples);
        for (a, b) in f32_samples.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_resample_i16() {
        let input: Vec<i16> = (0..160)
            .map(|i| ((i as f32 / 160.0) * 65534.0 - 32767.0) as i16)
            .collect();
        let output = resample_linear_i16(&input, 8000, 48000);

        assert!(output.len() >= 800);
    }

    #[test]
    fn test_resample_small_chunk() {
        // Small chunks should use linear fallback
        let input: Vec<f32> = vec![0.5; 32];
        let output = resample_linear(&input, 8000, 48000);

        assert!(!output.is_empty());
    }

    #[test]
    fn test_f32_to_i16_clipping() {
        // Test values beyond [-1, 1] range are clamped
        let input = vec![1.5, -1.5, 2.0, -2.0];
        let output = f32_to_i16(&input);

        assert_eq!(output[0], i16::MAX);
        assert_eq!(output[1], i16::MIN + 1); // -1.0 * 32767 = -32767
        assert_eq!(output[2], i16::MAX);
        assert_eq!(output[3], i16::MIN + 1);
    }

    #[test]
    fn test_i16_to_f32_range() {
        let input = vec![i16::MIN, 0, i16::MAX];
        let output = i16_to_f32(&input);

        assert!(output[0] <= -0.99 && output[0] >= -1.0);
        assert!((output[1]).abs() < 0.001);
        assert!(output[2] >= 0.99 && output[2] <= 1.0);
    }

    #[test]
    fn test_resample_voice_quality() {
        // Test typical voice path: 48kHz -> 8kHz -> 48kHz
        // Should preserve general shape even if not perfect
        let original: Vec<f32> = (0..960)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI / 960.0).sin())
            .collect();

        let downsampled = resample_linear(&original, 48000, 8000);
        let upsampled = resample_linear(&downsampled, 8000, 48000);

        // Check that the result has similar length (within reasonable bounds)
        assert!(
            upsampled.len() >= 800 && upsampled.len() <= 1100,
            "Got {} samples",
            upsampled.len()
        );

        // Check that the signal is not completely destroyed
        // (sum of absolute values should be similar)
        let orig_energy: f32 = original.iter().map(|x| x.abs()).sum();
        let result_energy: f32 = upsampled.iter().map(|x| x.abs()).sum();

        // Allow for some energy loss due to resampling, but should preserve most
        let energy_ratio = result_energy / orig_energy;
        assert!(
            energy_ratio > 0.8 && energy_ratio < 1.2,
            "Energy ratio {} is out of expected range (expected ~1.0)",
            energy_ratio
        );
    }

    #[test]
    fn test_resample_typical_voip_path() {
        // Simulate typical VoIP: 48kHz mic input → 8kHz codec
        // 20ms of audio at 48kHz = 960 samples
        let mic_frame: Vec<f32> = (0..960)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48000.0).sin() * 0.5)
            .collect();

        let codec_frame = resample_linear(&mic_frame, 48000, 8000);

        // 20ms at 8kHz = 160 samples
        assert!(
            codec_frame.len() >= 150 && codec_frame.len() <= 180,
            "Expected ~160 samples, got {}",
            codec_frame.len()
        );
    }
}

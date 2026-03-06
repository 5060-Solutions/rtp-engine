//! Audio resampling utilities.
//!
//! Provides sample rate conversion between codec rates (typically 8kHz)
//! and device rates (typically 44.1kHz or 48kHz).

/// Resample audio from one sample rate to another using linear interpolation.
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
pub fn resample_linear_i16(input: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
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

        let s0 = input.get(src_idx).copied().unwrap_or(0) as f64;
        let s1 = input
            .get(src_idx + 1)
            .copied()
            .unwrap_or_else(|| input.get(src_idx).copied().unwrap_or(0)) as f64;

        let sample = s0 + frac * (s1 - s0);
        output.push(sample.round() as i16);
    }

    output
}

/// Convert f32 samples (-1.0 to 1.0) to i16 samples.
pub fn f32_to_i16(input: &[f32]) -> Vec<i16> {
    input.iter().map(|&s| (s * 32767.0) as i16).collect()
}

/// Convert i16 samples to f32 samples (-1.0 to 1.0).
pub fn i16_to_f32(input: &[i16]) -> Vec<f32> {
    input.iter().map(|&s| s as f32 / 32768.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_upsample() {
        // Upsample 8kHz to 48kHz (6x)
        let input: Vec<f32> = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let output = resample_linear(&input, 8000, 48000);

        // Should have roughly 6x more samples
        assert!(output.len() >= 25);
        assert!(output.len() <= 35);
    }

    #[test]
    fn test_resample_downsample() {
        // Downsample 48kHz to 8kHz (1/6x)
        let input: Vec<f32> = (0..960).map(|i| (i as f32 / 960.0) * 2.0 - 1.0).collect();
        let output = resample_linear(&input, 48000, 8000);

        // Should have roughly 1/6 the samples
        assert!(output.len() >= 150);
        assert!(output.len() <= 180);
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
        let input: Vec<i16> = vec![0, 16384, 32767, 16384, 0];
        let output = resample_linear_i16(&input, 8000, 48000);

        assert!(output.len() >= 25);
    }

    #[test]
    fn test_resample_single_sample() {
        let input: Vec<f32> = vec![0.5];
        let output = resample_linear(&input, 8000, 48000);

        // Single sample replicates based on ratio
        assert!(!output.is_empty());
        assert!(output.iter().all(|&v| (v - 0.5).abs() < 0.001));
    }

    #[test]
    fn test_resample_boundary_values() {
        let input: Vec<f32> = vec![-1.0, 1.0, -1.0, 1.0];
        let output = resample_linear(&input, 8000, 16000);

        // Check values stay within [-1, 1]
        for v in &output {
            assert!(*v >= -1.0 && *v <= 1.0);
        }
    }

    #[test]
    fn test_f32_to_i16_clipping() {
        // Test values beyond [-1, 1] range
        let input = vec![1.5, -1.5, 2.0, -2.0];
        let output = f32_to_i16(&input);

        // Should clip to i16::MAX and i16::MIN
        assert_eq!(output[0], i16::MAX);
        assert_eq!(output[1], i16::MIN);
        assert_eq!(output[2], i16::MAX);
        assert_eq!(output[3], i16::MIN);
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
    fn test_resample_various_ratios() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 2.0 - 1.0).collect();

        // Test various common sample rate conversions
        let rates = [
            (8000, 16000),
            (16000, 8000),
            (8000, 44100),
            (44100, 8000),
            (48000, 44100),
            (44100, 48000),
        ];

        for (from, to) in rates {
            let output = resample_linear(&input, from, to);
            let expected_len = (input.len() as f64 * to as f64 / from as f64) as usize;
            // Allow some tolerance for rounding
            assert!(
                (output.len() as i64 - expected_len as i64).abs() <= 1,
                "Resample {}->{}: expected ~{}, got {}",
                from,
                to,
                expected_len,
                output.len()
            );
        }
    }

    #[test]
    fn test_resample_preserves_dc() {
        // DC signal (constant value) should remain constant
        let input: Vec<f32> = vec![0.5; 100];
        let output = resample_linear(&input, 8000, 48000);

        for v in &output {
            assert!((v - 0.5).abs() < 0.001, "DC not preserved: {}", v);
        }
    }
}

//! Audio call recording.
//!
//! Provides functionality to record both transmitted (TX) and received (RX) audio
//! during a call. Recordings are saved as WAV files.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Audio recorder that captures call audio to WAV files.
///
/// Records both directions (TX and RX) into separate channels of a stereo WAV file,
/// or as separate mono files.
pub struct CallRecorder {
    recording: Arc<AtomicBool>,
    tx_samples: Arc<Mutex<Vec<i16>>>,
    rx_samples: Arc<Mutex<Vec<i16>>>,
    output_path: PathBuf,
    sample_rate: u32,
}

impl CallRecorder {
    /// Create a new call recorder.
    ///
    /// # Arguments
    /// * `output_path` - Path for the output WAV file
    /// * `sample_rate` - Sample rate (typically 8000 for VoIP)
    pub fn new(output_path: PathBuf, sample_rate: u32) -> Self {
        Self {
            recording: Arc::new(AtomicBool::new(false)),
            tx_samples: Arc::new(Mutex::new(Vec::new())),
            rx_samples: Arc::new(Mutex::new(Vec::new())),
            output_path,
            sample_rate,
        }
    }

    /// Start recording.
    pub fn start(&self) {
        self.recording.store(true, Ordering::SeqCst);
        log::info!("Call recording started: {}", self.output_path.display());
    }

    /// Stop recording and write to file.
    pub fn stop(&self) -> Result<PathBuf, std::io::Error> {
        self.recording.store(false, Ordering::SeqCst);
        self.write_wav()
    }

    /// Check if currently recording.
    pub fn is_recording(&self) -> bool {
        self.recording.load(Ordering::SeqCst)
    }

    /// Get a handle for recording TX (transmitted/microphone) audio.
    pub fn tx_handle(&self) -> RecorderHandle {
        RecorderHandle {
            recording: self.recording.clone(),
            samples: self.tx_samples.clone(),
        }
    }

    /// Get a handle for recording RX (received/speaker) audio.
    pub fn rx_handle(&self) -> RecorderHandle {
        RecorderHandle {
            recording: self.recording.clone(),
            samples: self.rx_samples.clone(),
        }
    }

    /// Write recorded audio to WAV file.
    fn write_wav(&self) -> Result<PathBuf, std::io::Error> {
        let tx_samples = self.tx_samples.lock().unwrap();
        let rx_samples = self.rx_samples.lock().unwrap();

        // If no audio recorded, return early
        if tx_samples.is_empty() && rx_samples.is_empty() {
            log::warn!("No audio recorded, skipping WAV file creation");
            return Ok(self.output_path.clone());
        }

        // Create parent directory if needed
        if let Some(parent) = self.output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Determine the length (pad shorter channel with silence)
        let max_len = tx_samples.len().max(rx_samples.len());

        // Create stereo interleaved samples (TX on left, RX on right)
        let mut stereo_samples = Vec::with_capacity(max_len * 2);
        for i in 0..max_len {
            let tx = tx_samples.get(i).copied().unwrap_or(0);
            let rx = rx_samples.get(i).copied().unwrap_or(0);
            stereo_samples.push(tx);
            stereo_samples.push(rx);
        }

        // Write WAV file
        let file = File::create(&self.output_path)?;
        let mut writer = BufWriter::new(file);

        // WAV header
        let num_channels: u16 = 2;
        let bits_per_sample: u16 = 16;
        let byte_rate = self.sample_rate * u32::from(num_channels) * u32::from(bits_per_sample / 8);
        let block_align = num_channels * (bits_per_sample / 8);
        let data_size = (stereo_samples.len() * 2) as u32;
        let file_size = 36 + data_size;

        // RIFF header
        writer.write_all(b"RIFF")?;
        writer.write_all(&file_size.to_le_bytes())?;
        writer.write_all(b"WAVE")?;

        // fmt chunk
        writer.write_all(b"fmt ")?;
        writer.write_all(&16u32.to_le_bytes())?; // chunk size
        writer.write_all(&1u16.to_le_bytes())?; // PCM format
        writer.write_all(&num_channels.to_le_bytes())?;
        writer.write_all(&self.sample_rate.to_le_bytes())?;
        writer.write_all(&byte_rate.to_le_bytes())?;
        writer.write_all(&block_align.to_le_bytes())?;
        writer.write_all(&bits_per_sample.to_le_bytes())?;

        // data chunk
        writer.write_all(b"data")?;
        writer.write_all(&data_size.to_le_bytes())?;

        // Write samples
        for sample in stereo_samples {
            writer.write_all(&sample.to_le_bytes())?;
        }

        writer.flush()?;

        let duration_secs = max_len as f64 / self.sample_rate as f64;
        log::info!(
            "Call recording saved: {} ({:.1}s, {} samples)",
            self.output_path.display(),
            duration_secs,
            max_len
        );

        Ok(self.output_path.clone())
    }

    /// Get the output path.
    pub fn output_path(&self) -> &PathBuf {
        &self.output_path
    }
}

impl Drop for CallRecorder {
    fn drop(&mut self) {
        if self.recording.load(Ordering::SeqCst) {
            let _ = self.stop();
        }
    }
}

/// Handle for adding samples to a recorder from audio callbacks.
#[derive(Clone)]
pub struct RecorderHandle {
    recording: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<i16>>>,
}

impl RecorderHandle {
    /// Add samples if recording is active.
    pub fn add_samples(&self, samples: &[i16]) {
        if self.recording.load(Ordering::Relaxed) {
            if let Ok(mut buffer) = self.samples.lock() {
                buffer.extend_from_slice(samples);
            }
        }
    }

    /// Add f32 samples (will be converted to i16).
    pub fn add_samples_f32(&self, samples: &[f32]) {
        if self.recording.load(Ordering::Relaxed) {
            let i16_samples: Vec<i16> = samples
                .iter()
                .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                .collect();
            self.add_samples(&i16_samples);
        }
    }

    /// Check if recording is active.
    pub fn is_recording(&self) -> bool {
        self.recording.load(Ordering::Relaxed)
    }
}

/// Generate a recording filename with timestamp.
pub fn generate_recording_filename(call_id: &str, recordings_dir: &std::path::Path) -> PathBuf {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let safe_call_id: String = call_id
        .chars()
        .take(20)
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    recordings_dir.join(format!("call_{}_{}.wav", timestamp, safe_call_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_recorder_creation() {
        let path = PathBuf::from("/tmp/test_recording.wav");
        let recorder = CallRecorder::new(path.clone(), 8000);

        assert!(!recorder.is_recording());
        assert_eq!(recorder.output_path(), &path);
    }

    #[test]
    fn test_recorder_handle() {
        let path = PathBuf::from("/tmp/test_handle.wav");
        let recorder = CallRecorder::new(path, 8000);

        let tx_handle = recorder.tx_handle();
        let rx_handle = recorder.rx_handle();

        // Not recording yet
        assert!(!tx_handle.is_recording());

        recorder.start();
        assert!(tx_handle.is_recording());
        assert!(rx_handle.is_recording());

        // Add samples
        tx_handle.add_samples(&[100, 200, 300]);
        rx_handle.add_samples(&[-100, -200, -300]);

        // Stop doesn't fail
        let _ = recorder.stop();
    }

    #[test]
    fn test_recorder_write_wav() {
        let dir = std::env::temp_dir().join("rtp_engine_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_write.wav");

        let recorder = CallRecorder::new(path.clone(), 8000);
        recorder.start();

        let tx = recorder.tx_handle();
        let rx = recorder.rx_handle();

        // Add a short sine wave
        let samples: Vec<i16> = (0..800)
            .map(|i| ((i as f64 * 0.1).sin() * 16000.0) as i16)
            .collect();

        tx.add_samples(&samples);
        rx.add_samples(&samples);

        let result = recorder.stop();
        assert!(result.is_ok());

        // Verify file exists and has reasonable size
        let metadata = fs::metadata(&path);
        assert!(metadata.is_ok());
        assert!(metadata.unwrap().len() > 44); // At least WAV header

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_generate_filename() {
        let dir = PathBuf::from("/recordings");
        let filename = generate_recording_filename("test-call-123", &dir);

        assert!(filename.to_string_lossy().contains("call_"));
        assert!(filename.to_string_lossy().ends_with(".wav"));
        assert!(filename.starts_with(&dir));
    }

    #[test]
    fn test_recorder_empty() {
        let dir = std::env::temp_dir().join("rtp_engine_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_empty.wav");

        let recorder = CallRecorder::new(path.clone(), 8000);
        recorder.start();
        // Don't add any samples
        let result = recorder.stop();

        // Should succeed without creating file (or creating empty file)
        assert!(result.is_ok());
    }
}

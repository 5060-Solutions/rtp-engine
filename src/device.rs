//! Audio device abstraction for capture and playback.
//!
//! Provides cross-platform audio I/O using cpal, with automatic resampling
//! between device rates and codec rates (8kHz for G.711).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::resample::{f32_to_i16, i16_to_f32, resample_linear};

/// Audio capture device (microphone).
pub struct AudioCapture {
    _stream: cpal::Stream,
    running: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
    device_rate: u32,
}

impl AudioCapture {
    /// Start capturing audio from the default input device.
    ///
    /// Samples are buffered internally and can be retrieved with `read_samples()`.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| Error::device("No input audio device"))?;

        let config = device
            .default_input_config()
            .map_err(|e| Error::device(format!("No default input config: {}", e)))?;

        let device_rate = config.sample_rate();
        log::info!("Audio capture: device rate = {} Hz", device_rate);

        let stream_config = cpal::StreamConfig {
            channels: 1,
            sample_rate: config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let running = Arc::new(AtomicBool::new(true));
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(8192)));

        let cb_running = running.clone();
        let cb_buffer = buffer.clone();

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !cb_running.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Ok(mut buf) = cb_buffer.lock() {
                        buf.extend_from_slice(data);
                        // Limit buffer size to ~1 second
                        while buf.len() > device_rate as usize {
                            buf.drain(..device_rate as usize / 10);
                        }
                    }
                },
                |err| log::error!("Audio capture error: {}", err),
                None,
            )
            .map_err(|e| Error::device(format!("Failed to build input stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| Error::device(format!("Failed to start capture: {}", e)))?;

        Ok(Self {
            _stream: stream,
            running,
            buffer,
            device_rate,
        })
    }

    /// Read samples from the capture buffer, resampled to the target rate.
    ///
    /// Returns up to `max_samples` samples at the target sample rate.
    pub fn read_samples(&self, target_rate: u32, max_samples: usize) -> Vec<i16> {
        let mut result = Vec::new();

        if let Ok(mut buf) = self.buffer.lock() {
            if buf.is_empty() {
                return result;
            }

            // Calculate how many device samples we need for the requested output
            let device_samples_needed = ((max_samples as f64)
                * (self.device_rate as f64 / target_rate as f64))
                .ceil() as usize;
            let available = buf.len().min(device_samples_needed);

            if available > 0 {
                let samples: Vec<f32> = buf.drain(..available).collect();
                let resampled = resample_linear(&samples, self.device_rate, target_rate);
                result = f32_to_i16(&resampled);
            }
        }

        result
    }

    /// Get the native device sample rate.
    pub fn device_rate(&self) -> u32 {
        self.device_rate
    }

    /// Stop capturing.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Audio playback device (speaker).
pub struct AudioPlayback {
    _stream: cpal::Stream,
    running: Arc<AtomicBool>,
    buffer: Arc<Mutex<VecDeque<f32>>>,
    device_rate: u32,
}

impl AudioPlayback {
    /// Start audio playback to the default output device.
    ///
    /// Samples can be written with `write_samples()`.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| Error::device("No output audio device"))?;

        let config = device
            .default_output_config()
            .map_err(|e| Error::device(format!("No default output config: {}", e)))?;

        let device_rate = config.sample_rate();
        log::info!("Audio playback: device rate = {} Hz", device_rate);

        let stream_config = cpal::StreamConfig {
            channels: 1,
            sample_rate: config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let running = Arc::new(AtomicBool::new(true));
        let buffer: Arc<Mutex<VecDeque<f32>>> =
            Arc::new(Mutex::new(VecDeque::with_capacity(device_rate as usize)));

        let cb_buffer = buffer.clone();

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if let Ok(mut buf) = cb_buffer.lock() {
                        for sample in data.iter_mut() {
                            *sample = buf.pop_front().unwrap_or(0.0);
                        }
                    } else {
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                    }
                },
                |err| log::error!("Audio playback error: {}", err),
                None,
            )
            .map_err(|e| Error::device(format!("Failed to build output stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| Error::device(format!("Failed to start playback: {}", e)))?;

        Ok(Self {
            _stream: stream,
            running,
            buffer,
            device_rate,
        })
    }

    /// Write samples to the playback buffer.
    ///
    /// Samples are resampled from the source rate to the device rate.
    pub fn write_samples(&self, samples: &[i16], source_rate: u32) {
        let f32_samples = i16_to_f32(samples);
        let resampled = resample_linear(&f32_samples, source_rate, self.device_rate);

        if let Ok(mut buf) = self.buffer.lock() {
            for s in resampled {
                buf.push_back(s);
            }
            // Limit buffer size to ~1 second
            while buf.len() > self.device_rate as usize {
                buf.pop_front();
            }
        }
    }

    /// Get the native device sample rate.
    pub fn device_rate(&self) -> u32 {
        self.device_rate
    }

    /// Stop playback.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Drop for AudioPlayback {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Query available audio devices.
pub fn list_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(desc) = device.description() {
                devices.push(format!("Input: {}", desc.name()));
            }
        }
    }

    if let Ok(output_devices) = host.output_devices() {
        for device in output_devices {
            if let Ok(desc) = device.description() {
                devices.push(format!("Output: {}", desc.name()));
            }
        }
    }

    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_devices() {
        // This test may fail in CI without audio devices, but should work locally
        let result = list_devices();
        assert!(result.is_ok());
    }
}

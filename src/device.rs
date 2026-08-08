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

/// Create cpal's cached device enumerator on a thread that never exits.
///
/// cpal caches one `IMMDeviceEnumerator` in a process-wide `OnceLock`, created
/// on whichever thread asks first, and marks it `Send + Sync`. But the COM
/// initialisation backing it is a *thread-local* whose `Drop` calls
/// `CoUninitialize`. So when the creating thread exits, COM tears down its
/// apartment and the cached pointer dangles — while every later call keeps
/// reading its vtable. That is the `INVALID_POINTER_READ_c0000005` seen at
/// `+0x19d` in both `EnumAudioEndpoints` and `GetDefaultAudioEndpoint`: the same
/// dead pointer reached by two paths.
///
/// It matters here because audio threads are per-call. Without this, the first
/// call to finish would invalidate audio device access for the rest of the
/// process, and a later call would fault.
///
/// Priming from a parked thread means the apartment outlives every caller. The
/// thread is deliberately never joined; one parked thread for the life of the
/// process is the cost of a valid enumerator.
///
/// No-op away from Windows, where none of this applies.
fn prime_device_enumerator() {
    static PRIMED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

    if !cfg!(windows) {
        return;
    }

    PRIMED.get_or_init(|| {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("rtp-engine-com".to_owned())
            .spawn(move || {
                // Touching the enumeration is what constructs cpal's cached
                // enumerator; the result is irrelevant.
                let host = cpal::default_host();
                let _ = host.input_devices().map(Iterator::count);
                let _ = ready_tx.send(());
                loop {
                    std::thread::park();
                }
            });

        if let Err(e) = spawned {
            log::warn!("Could not start the audio COM thread: {e}");
            return;
        }
        // Wait for the enumerator to exist before any caller uses it. If the
        // thread died first, carry on: the call was going to fault either way,
        // and blocking forever here would be worse.
        if ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .is_err()
        {
            log::warn!("Audio COM thread did not report ready; continuing anyway");
        }
    });
}

/// `Host::default_input_device`, but returns `None` instead of asking for a
/// default that does not exist.
///
/// Enumerating first is only meaningful once [`prime_device_enumerator`] has
/// made the enumerator outlive its creator; before that, both this and
/// enumeration read the same dangling pointer.
pub(crate) fn safe_default_input_device(host: &cpal::Host) -> Option<cpal::Device> {
    prime_device_enumerator();
    if !has_any(host.input_devices()) {
        log::warn!("No audio input devices present; not requesting a default one");
        return None;
    }
    host.default_input_device()
}

/// `Host::default_output_device`, with the same guard as
/// [`safe_default_input_device`].
pub(crate) fn safe_default_output_device(host: &cpal::Host) -> Option<cpal::Device> {
    prime_device_enumerator();
    if !has_any(host.output_devices()) {
        log::warn!("No audio output devices present; not requesting a default one");
        return None;
    }
    host.default_output_device()
}

/// Whether an enumeration yielded at least one device.
///
/// A failed enumeration counts as none: if the host cannot list its devices,
/// asking it for a default is not going to go better.
fn has_any<D>(devices: std::result::Result<D, cpal::DevicesError>) -> bool
where
    D: Iterator<Item = cpal::Device>,
{
    devices.is_ok_and(|mut d| d.next().is_some())
}

/// Audio capture device (microphone).
pub struct AudioCapture {
    _stream: cpal::Stream,
    running: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
    device_rate: u32,
}

impl AudioCapture {
    /// Start capturing audio from a named input device.
    ///
    /// If `device_name` is None, uses the system default.
    pub fn start_with_device_name(device_name: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => {
                let mut found = None;
                if let Ok(devices) = host.input_devices() {
                    for d in devices {
                        if let Ok(desc) = d.description() {
                            if desc.name() == name {
                                found = Some(d);
                                break;
                            }
                        }
                    }
                }
                found.or_else(|| {
                    log::warn!("Input device '{}' not found, using default", name);
                    safe_default_input_device(&host)
                })
            }
            None => safe_default_input_device(&host),
        }
        .ok_or_else(|| Error::device("No input audio device"))?;

        Self::start_from_device(device)
    }

    /// Start capturing audio from the default input device.
    ///
    /// Samples are buffered internally and can be retrieved with `read_samples()`.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = safe_default_input_device(&host)
            .ok_or_else(|| Error::device("No input audio device"))?;

        Self::start_from_device(device)
    }

    fn start_from_device(device: cpal::Device) -> Result<Self> {
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
    /// Start audio playback to a named output device.
    ///
    /// If `device_name` is None, uses the system default.
    pub fn start_with_device_name(device_name: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => {
                let mut found = None;
                if let Ok(devices) = host.output_devices() {
                    for d in devices {
                        if let Ok(desc) = d.description() {
                            if desc.name() == name {
                                found = Some(d);
                                break;
                            }
                        }
                    }
                }
                found.or_else(|| {
                    log::warn!("Output device '{}' not found, using default", name);
                    safe_default_output_device(&host)
                })
            }
            None => safe_default_output_device(&host),
        }
        .ok_or_else(|| Error::device("No output audio device"))?;

        Self::start_from_device(device)
    }

    /// Start audio playback to the default output device.
    ///
    /// Samples can be written with `write_samples()`.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = safe_default_output_device(&host)
            .ok_or_else(|| Error::device("No output audio device"))?;

        Self::start_from_device(device)
    }

    fn start_from_device(device: cpal::Device) -> Result<Self> {
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

/// Information about an audio device.
#[derive(Debug, Clone)]
pub struct AudioDevice {
    /// Device name/identifier
    pub name: String,
    /// Whether this is the default device
    pub is_default: bool,
}

/// Lists of available audio devices.
#[derive(Debug, Clone, Default)]
pub struct AudioDevices {
    /// Available input (microphone) devices
    pub input_devices: Vec<AudioDevice>,
    /// Available output (speaker) devices
    pub output_devices: Vec<AudioDevice>,
}

/// Query available audio input devices (microphones).
pub fn list_input_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let default_name = safe_default_input_device(&host)
        .and_then(|d| d.description().ok())
        .map(|d| d.name().to_string());

    let mut devices = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(desc) = device.description() {
                let name = desc.name().to_string();
                let is_default = default_name.as_ref() == Some(&name);
                devices.push(AudioDevice { name, is_default });
            }
        }
    }
    Ok(devices)
}

/// Query available audio output devices (speakers).
pub fn list_output_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let default_name = safe_default_output_device(&host)
        .and_then(|d| d.description().ok())
        .map(|d| d.name().to_string());

    let mut devices = Vec::new();
    if let Ok(output_devices) = host.output_devices() {
        for device in output_devices {
            if let Ok(desc) = device.description() {
                let name = desc.name().to_string();
                let is_default = default_name.as_ref() == Some(&name);
                devices.push(AudioDevice { name, is_default });
            }
        }
    }
    Ok(devices)
}

/// Query all available audio devices (both input and output).
pub fn list_all_devices() -> Result<AudioDevices> {
    Ok(AudioDevices {
        input_devices: list_input_devices()?,
        output_devices: list_output_devices()?,
    })
}

/// Query available audio devices (legacy - returns combined list).
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

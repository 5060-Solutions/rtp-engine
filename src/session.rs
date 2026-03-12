//! Media session management.
//!
//! The `MediaSession` struct orchestrates the complete audio pipeline:
//! microphone capture → encoding → RTP transmission → reception → decoding → speaker playback.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::net::UdpSocket;

use crate::codec::CodecType;
#[cfg(feature = "device")]
use crate::codec::{AudioDecoder, AudioEncoder, create_decoder, create_encoder};
use crate::error::{Error, Result};
use crate::recorder::{CallRecorder, RecorderHandle};
#[cfg(feature = "device")]
use crate::resample::{StreamResampler, f32_to_i16, i16_to_f32, resample_linear};
#[cfg(feature = "device")]
use crate::rtp::RtpHeader;
use crate::rtp::{RtpCounters, RtpStats, build_rtcp_rr, build_rtcp_sr};
#[cfg(feature = "device")]
use crate::rtp::{parse_rtp, parse_sequence, parse_timestamp};

#[cfg(feature = "srtp")]
use crate::srtp::SrtpContext;

#[cfg(feature = "device")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

type SharedSrtp = Arc<std::sync::Mutex<SrtpContext>>;

/// A complete RTP media session with bidirectional audio.
///
/// Manages:
/// - Audio capture from microphone
/// - Encoding with the configured codec
/// - RTP packet transmission
/// - RTP packet reception
/// - Decoding and playback to speaker
/// - RTCP statistics reporting
/// - Optional call recording
pub struct MediaSession {
    muted: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    counters: RtpCounters,
    codec: CodecType,
    learned_remote: Arc<std::sync::Mutex<Option<SocketAddr>>>,
    rtp_socket: Arc<UdpSocket>,
    ssrc: u32,
    remote_addr: SocketAddr,
    recorder: Arc<std::sync::Mutex<Option<CallRecorder>>>,
    tx_recorder_handle: Arc<std::sync::Mutex<Option<RecorderHandle>>>,
    rx_recorder_handle: Arc<std::sync::Mutex<Option<RecorderHandle>>>,
    /// Conference mode flag - when enabled, audio from this session
    /// participates in multi-party mixing
    conference_mode: Arc<AtomicBool>,
}

impl MediaSession {
    /// Allocate an available RTP port pair by letting the OS choose.
    ///
    /// Binds to port 0 to get an OS-assigned port, then verifies the adjacent
    /// RTCP port (RTP+1) is also available. Returns the RTP port number.
    ///
    /// This is the recommended way to get a port before building SDP, as it
    /// guarantees the port is available at allocation time.
    ///
    /// # Example
    /// ```ignore
    /// let rtp_port = MediaSession::allocate_port().await?;
    /// // Use rtp_port in SDP...
    /// // Later, start the session with that port
    /// let session = MediaSession::start(rtp_port, remote, codec).await?;
    /// ```
    pub async fn allocate_port() -> Result<u16> {
        // Bind RTP socket to port 0 to get OS-assigned port
        let rtp_socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| Error::Network(e))?;

        let rtp_port = rtp_socket
            .local_addr()
            .map_err(|e| Error::Network(e))?
            .port();

        // Make sure port is even (RTP convention) - if odd, try again
        if rtp_port % 2 != 0 {
            drop(rtp_socket);
            return Box::pin(Self::allocate_port()).await;
        }

        // Try to bind RTCP port (RTP + 1) to verify it's available
        let rtcp_port = rtp_port + 1;
        let rtcp_socket = UdpSocket::bind(format!("0.0.0.0:{}", rtcp_port)).await;

        if rtcp_socket.is_err() {
            // RTCP port not available, try again
            drop(rtp_socket);
            return Box::pin(Self::allocate_port()).await;
        }

        // Both ports verified available - drop sockets and return the port
        // The port will be re-bound when start() is called
        // Note: There's a small race window here, but it's acceptable
        drop(rtcp_socket);
        drop(rtp_socket);

        Ok(rtp_port)
    }

    /// Start a media session with the specified codec.
    ///
    /// # Arguments
    /// * `local_rtp_port` - Local UDP port for RTP (use 0 for OS-assigned, or
    ///   use a port from `allocate_port()` for guaranteed availability)
    /// * `remote_addr` - Remote RTP endpoint address
    /// * `codec_type` - Audio codec to use
    pub async fn start(
        local_rtp_port: u16,
        remote_addr: SocketAddr,
        codec_type: CodecType,
    ) -> Result<Self> {
        Self::start_internal(local_rtp_port, remote_addr, codec_type, None, None).await
    }

    /// Start a media session with SRTP encryption (single key for both directions).
    ///
    /// Note: For SDES key exchange where each party has their own key, use
    /// `start_with_srtp_keys` instead.
    #[cfg(feature = "srtp")]
    pub async fn start_with_srtp(
        local_rtp_port: u16,
        remote_addr: SocketAddr,
        codec_type: CodecType,
        srtp_ctx: SrtpContext,
    ) -> Result<Self> {
        Self::start_internal(
            local_rtp_port,
            remote_addr,
            codec_type,
            Some(srtp_ctx),
            None,
        )
        .await
    }

    /// Start a media session with SRTP using separate keys for TX and RX.
    ///
    /// In SDES key exchange (RFC 4568), each party generates their own key:
    /// - `tx_srtp_ctx`: Our key (from our SDP offer) for encrypting outgoing packets
    /// - `rx_srtp_ctx`: Their key (from their SDP answer) for decrypting incoming packets
    #[cfg(feature = "srtp")]
    pub async fn start_with_srtp_keys(
        local_rtp_port: u16,
        remote_addr: SocketAddr,
        codec_type: CodecType,
        tx_srtp_ctx: SrtpContext,
        rx_srtp_ctx: SrtpContext,
    ) -> Result<Self> {
        Self::start_internal(
            local_rtp_port,
            remote_addr,
            codec_type,
            Some(tx_srtp_ctx),
            Some(rx_srtp_ctx),
        )
        .await
    }

    async fn start_internal(
        local_rtp_port: u16,
        remote_addr: SocketAddr,
        codec_type: CodecType,
        #[allow(unused_variables)] tx_srtp_ctx: Option<SrtpContext>,
        #[allow(unused_variables)] rx_srtp_ctx: Option<SrtpContext>,
    ) -> Result<Self> {
        let rtp_socket = UdpSocket::bind(format!("0.0.0.0:{}", local_rtp_port))
            .await
            .map_err(|e| Error::Network(e))?;

        let rtp_socket = Arc::new(rtp_socket);
        let muted = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(true));
        let ssrc: u32 = rand::random();
        let counters = RtpCounters::new(codec_type.name());
        let learned_remote: Arc<std::sync::Mutex<Option<SocketAddr>>> =
            Arc::new(std::sync::Mutex::new(None));

        // Recorder handles (will be populated when recording starts)
        let tx_recorder_handle: Arc<std::sync::Mutex<Option<RecorderHandle>>> =
            Arc::new(std::sync::Mutex::new(None));
        let rx_recorder_handle: Arc<std::sync::Mutex<Option<RecorderHandle>>> =
            Arc::new(std::sync::Mutex::new(None));

        #[cfg(feature = "device")]
        let encoder = create_encoder(codec_type)?;
        #[cfg(feature = "device")]
        let decoder = create_decoder(codec_type)?;

        // SRTP contexts: TX for outgoing (protect), RX for incoming (unprotect)
        // If only tx_srtp_ctx is provided, use it for both directions (symmetric mode)
        #[cfg(feature = "srtp")]
        let (tx_srtp, rx_srtp): (Option<SharedSrtp>, Option<SharedSrtp>) =
            match (tx_srtp_ctx, rx_srtp_ctx) {
                (Some(tx), Some(rx)) => {
                    log::info!("MediaSession: SRTP enabled with separate TX and RX contexts");
                    (
                        Some(Arc::new(std::sync::Mutex::new(tx))),
                        Some(Arc::new(std::sync::Mutex::new(rx))),
                    )
                }
                (Some(tx), None) => {
                    log::info!("MediaSession: SRTP enabled with symmetric mode (shared context)");
                    let shared = Arc::new(std::sync::Mutex::new(tx));
                    (Some(shared.clone()), Some(shared))
                }
                _ => {
                    log::info!("MediaSession: No SRTP - using plain RTP");
                    (None, None)
                }
            };
        #[cfg(not(feature = "srtp"))]
        let (tx_srtp, rx_srtp): (Option<SharedSrtp>, Option<SharedSrtp>) = (None, None);

        // RTCP socket (RTP port + 1)
        let rtcp_port = local_rtp_port + 1;
        let rtcp_socket = UdpSocket::bind(format!("0.0.0.0:{}", rtcp_port))
            .await
            .map_err(|e| Error::Network(e))?;
        let rtcp_socket = Arc::new(rtcp_socket);
        let remote_rtcp_addr: SocketAddr =
            format!("{}:{}", remote_addr.ip(), remote_addr.port() + 1)
                .parse()
                .unwrap_or(remote_addr);

        // Start TX thread (microphone → RTP)
        #[cfg(feature = "device")]
        {
            let tx_socket = rtp_socket.clone();
            let tx_muted = muted.clone();
            let tx_running = running.clone();
            let tx_counters = counters.clone();
            let tx_learned = learned_remote.clone();
            let tx_srtp_ctx = tx_srtp.clone();
            let tx_recorder = tx_recorder_handle.clone();

            std::thread::spawn(move || {
                if let Err(e) = run_audio_tx(
                    tx_socket,
                    remote_addr,
                    ssrc,
                    tx_muted,
                    tx_running,
                    encoder,
                    tx_counters,
                    tx_learned,
                    tx_srtp_ctx,
                    tx_recorder,
                ) {
                    log::error!("Audio TX error: {}", e);
                }
            });
        }

        // Start RX thread (RTP → speaker)
        #[cfg(feature = "device")]
        {
            let rx_socket = rtp_socket.clone();
            let rx_running = running.clone();
            let rx_counters = counters.clone();
            let rx_learned = learned_remote.clone();
            let rx_srtp_ctx = rx_srtp.clone();
            let rx_recorder = rx_recorder_handle.clone();

            std::thread::spawn(move || {
                if let Err(e) = run_audio_rx(
                    rx_socket,
                    rx_running,
                    decoder,
                    rx_counters,
                    rx_learned,
                    rx_srtp_ctx,
                    rx_recorder,
                ) {
                    log::error!("Audio RX error: {}", e);
                }
            });
        }

        // Start RTCP task (uses TX context for protect, RX context for unprotect)
        {
            let rtcp_running = running.clone();
            let rtcp_counters = counters.clone();
            let rtcp_tx_srtp = tx_srtp;
            let rtcp_rx_srtp = rx_srtp;
            tokio::spawn(async move {
                run_rtcp(
                    rtcp_socket,
                    remote_rtcp_addr,
                    ssrc,
                    rtcp_running,
                    rtcp_counters,
                    rtcp_tx_srtp,
                    rtcp_rx_srtp,
                )
                .await;
            });
        }

        log::info!(
            "Media session started: local RTP :{}, remote {}, codec {:?}",
            local_rtp_port,
            remote_addr,
            codec_type,
        );

        Ok(Self {
            muted,
            running,
            counters,
            codec: codec_type,
            learned_remote,
            rtp_socket,
            ssrc,
            remote_addr,
            recorder: Arc::new(std::sync::Mutex::new(None)),
            tx_recorder_handle,
            rx_recorder_handle,
            conference_mode: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Send an RFC 2833 DTMF digit.
    pub fn send_dtmf(&self, digit: &str) {
        let event_code: u8 = match digit {
            "0" => 0,
            "1" => 1,
            "2" => 2,
            "3" => 3,
            "4" => 4,
            "5" => 5,
            "6" => 6,
            "7" => 7,
            "8" => 8,
            "9" => 9,
            "*" => 10,
            "#" => 11,
            _ => {
                log::warn!("Unknown DTMF digit: {}", digit);
                return;
            }
        };

        let socket = self.rtp_socket.clone();
        let ssrc = self.ssrc;
        let dest = self
            .learned_remote
            .lock()
            .ok()
            .and_then(|g| *g)
            .unwrap_or(self.remote_addr);
        let counters = self.counters.clone();

        tokio::spawn(async move {
            let base_ts: u32 = rand::random();
            let base_seq: u16 = rand::random();
            let volume: u8 = 10;
            let pt: u8 = 101;
            let durations = [160u16, 320, 480];

            for (i, &duration) in durations.iter().enumerate() {
                let is_end = i == durations.len() - 1;
                let seq = base_seq.wrapping_add(i as u16);

                let mut packet = Vec::with_capacity(16);
                packet.push(0x80);
                let marker = if i == 0 { 0x80 } else { 0x00 };
                packet.push(pt | marker);
                packet.extend_from_slice(&seq.to_be_bytes());
                packet.extend_from_slice(&base_ts.to_be_bytes());
                packet.extend_from_slice(&ssrc.to_be_bytes());

                let end_flag: u8 = if is_end { 0x80 } else { 0x00 };
                packet.push(event_code);
                packet.push(end_flag | (volume & 0x3F));
                packet.extend_from_slice(&duration.to_be_bytes());

                let _ = socket.send_to(&packet, dest).await;
                counters.record_sent(packet.len() as u64);

                if is_end {
                    for _ in 0..2 {
                        let repeat_seq = seq.wrapping_add(1);
                        packet[2..4].copy_from_slice(&repeat_seq.to_be_bytes());
                        let _ = socket.send_to(&packet, dest).await;
                    }
                }

                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        });
    }

    /// Set mute state.
    pub fn set_mute(&self, mute: bool) {
        self.muted.store(mute, Ordering::Relaxed);
    }

    /// Check if muted.
    pub fn is_muted(&self) -> bool {
        self.muted.load(Ordering::Relaxed)
    }

    /// Enable or disable conference mode for this session.
    /// In conference mode, audio from this session participates in multi-party mixing.
    pub fn set_conference_mode(&self, enabled: bool) {
        self.conference_mode.store(enabled, Ordering::Relaxed);
    }

    /// Check if conference mode is enabled.
    pub fn is_conference_mode(&self) -> bool {
        self.conference_mode.load(Ordering::Relaxed)
    }

    /// Get current statistics.
    pub fn stats(&self) -> RtpStats {
        self.counters.snapshot()
    }

    /// Get the codec in use.
    pub fn codec(&self) -> CodecType {
        self.codec
    }

    /// Get the SSRC.
    pub fn ssrc(&self) -> u32 {
        self.ssrc
    }

    /// Get the remote address.
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// Get the learned remote address (for symmetric RTP/comedia).
    pub fn learned_remote(&self) -> Option<SocketAddr> {
        self.learned_remote.lock().ok().and_then(|g| *g)
    }

    /// Start recording the call to a WAV file.
    ///
    /// The recording captures both TX (microphone) and RX (speaker) audio
    /// as a stereo WAV file with TX on the left channel and RX on the right.
    ///
    /// # Arguments
    /// * `output_path` - Path for the output WAV file
    pub fn start_recording(&self, output_path: std::path::PathBuf) -> Result<()> {
        let recorder = CallRecorder::new(output_path, 8000); // VoIP sample rate

        // Get handles for TX and RX
        let tx_handle = recorder.tx_handle();
        let rx_handle = recorder.rx_handle();

        // Start recording
        recorder.start();

        // Store handles for the audio threads to use
        if let Ok(mut handle) = self.tx_recorder_handle.lock() {
            *handle = Some(tx_handle);
        }
        if let Ok(mut handle) = self.rx_recorder_handle.lock() {
            *handle = Some(rx_handle);
        }

        // Store the recorder
        if let Ok(mut rec) = self.recorder.lock() {
            *rec = Some(recorder);
        }

        log::info!("Call recording started");
        Ok(())
    }

    /// Stop recording and save the WAV file.
    ///
    /// Returns the path to the saved recording.
    pub fn stop_recording(&self) -> Result<Option<std::path::PathBuf>> {
        // Clear handles first
        if let Ok(mut handle) = self.tx_recorder_handle.lock() {
            *handle = None;
        }
        if let Ok(mut handle) = self.rx_recorder_handle.lock() {
            *handle = None;
        }

        // Stop and save
        if let Ok(mut rec) = self.recorder.lock() {
            if let Some(recorder) = rec.take() {
                let path = recorder
                    .stop()
                    .map_err(|e| Error::device(format!("Recording save failed: {}", e)))?;
                log::info!("Call recording stopped and saved");
                return Ok(Some(path));
            }
        }

        Ok(None)
    }

    /// Check if recording is currently active.
    pub fn is_recording(&self) -> bool {
        self.recorder
            .lock()
            .ok()
            .and_then(|r| r.as_ref().map(super::recorder::CallRecorder::is_recording))
            .unwrap_or(false)
    }

    /// Get a clone of the TX recorder handle (for audio thread integration).
    pub fn get_tx_recorder_handle(&self) -> Option<RecorderHandle> {
        self.tx_recorder_handle.lock().ok().and_then(|h| h.clone())
    }

    /// Get a clone of the RX recorder handle (for audio thread integration).
    pub fn get_rx_recorder_handle(&self) -> Option<RecorderHandle> {
        self.rx_recorder_handle.lock().ok().and_then(|h| h.clone())
    }

    /// Stop the media session.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        log::info!("Media session stopped");
    }
}

impl Drop for MediaSession {
    fn drop(&mut self) {
        self.stop();
    }
}

impl std::fmt::Debug for MediaSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaSession")
            .field("codec", &self.codec)
            .field("ssrc", &self.ssrc)
            .field("remote_addr", &self.remote_addr)
            .field("muted", &self.muted.load(Ordering::Relaxed))
            .field("running", &self.running.load(Ordering::Relaxed))
            .finish()
    }
}

// --- Audio TX/RX implementation ---

#[cfg(feature = "device")]
fn run_audio_tx(
    socket: Arc<UdpSocket>,
    remote: SocketAddr,
    ssrc: u32,
    muted: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    encoder: Box<dyn AudioEncoder>,
    counters: RtpCounters,
    learned_remote: Arc<std::sync::Mutex<Option<SocketAddr>>>,
    _srtp: Option<SharedSrtp>,
    recorder_handle: Arc<std::sync::Mutex<Option<RecorderHandle>>>,
) -> Result<()> {
    use std::sync::atomic::AtomicU16;

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| Error::device("No input device"))?;

    let default_config = device
        .default_input_config()
        .map_err(|e| Error::device(format!("No input config: {}", e)))?;

    let native_rate = default_config.sample_rate();
    log::info!("Audio TX: native rate = {} Hz", native_rate);

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: default_config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let codec_rate = 8000u32;
    let resample_ratio = codec_rate as f64 / native_rate as f64;

    // Create a multi-threaded runtime for async UDP sends from audio callback
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .map_err(|e| Error::device(format!("Failed to create TX runtime: {}", e)))?;
    let rt_handle = rt.handle().clone();

    let seq = Arc::new(AtomicU16::new(0));
    let ts = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let pt = encoder.payload_type();
    let encoder = Arc::new(std::sync::Mutex::new(encoder));
    let sample_buffer = Arc::new(std::sync::Mutex::new(Vec::<f32>::with_capacity(1024)));
    let samples_per_frame = 160usize;
    let native_samples_per_frame = ((samples_per_frame as f64) / resample_ratio).ceil() as usize;

    // Persistent TX resampler: native rate → codec rate (e.g. 48kHz → 8kHz)
    let tx_resampler = Arc::new(std::sync::Mutex::new(StreamResampler::new(
        native_rate,
        codec_rate,
        native_samples_per_frame,
    )));

    // Silence PCM buffer for muted/keepalive packets (160 samples of silence)
    let silence_pcm: Vec<i16> = vec![0i16; samples_per_frame];

    let cb_running = running.clone();
    let cb_muted = muted.clone();
    let cb_seq = seq.clone();
    let cb_ts = ts.clone();
    let cb_encoder = encoder.clone();
    let cb_socket = socket.clone();
    let cb_learned = learned_remote.clone();
    let cb_counters = counters.clone();
    let cb_rt = rt_handle.clone();
    let cb_recorder = recorder_handle.clone();
    let cb_resampler = tx_resampler.clone();
    #[cfg(feature = "srtp")]
    let cb_srtp = _srtp.clone();

    // Counter for logging mic audio packets (only log periodically to avoid spam)
    let mic_packet_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let cb_mic_count = mic_packet_count.clone();
    let first_audio_logged = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cb_first_logged = first_audio_logged.clone();

    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !cb_running.load(Ordering::Relaxed) {
                    return;
                }

                // When muted, skip mic data processing but let keepalive task send silence
                if cb_muted.load(Ordering::Relaxed) {
                    return;
                }

                let mut buffer = match sample_buffer.lock() {
                    Ok(b) => b,
                    Err(_) => return,
                };
                buffer.extend_from_slice(data);

                while buffer.len() >= native_samples_per_frame {
                    let chunk: Vec<f32> = buffer.drain(..native_samples_per_frame).collect();

                    // Calculate audio level for debugging (RMS)
                    let rms: f32 =
                        (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();

                    // Use persistent streaming resampler for glitch-free audio
                    let resampled = match cb_resampler.lock() {
                        Ok(mut r) => r.process(&chunk),
                        Err(_) => resample_linear(&chunk, native_rate, codec_rate),
                    };
                    let pcm = f32_to_i16(&resampled);

                    // Record TX audio if recording is active
                    if let Ok(handle) = cb_recorder.lock() {
                        if let Some(ref rec) = *handle {
                            rec.add_samples(&pcm);
                        }
                    }

                    let current_seq = cb_seq.fetch_add(1, Ordering::Relaxed);
                    let current_ts = cb_ts.fetch_add(samples_per_frame as u32, Ordering::Relaxed);

                    let header = RtpHeader::new(pt, current_seq, current_ts, ssrc);
                    let mut packet = header.to_bytes();

                    if let Ok(mut enc) = cb_encoder.lock() {
                        enc.encode(&pcm, &mut packet);
                    }

                    #[cfg(feature = "srtp")]
                    let send_packet = if let Some(ref srtp_ctx) = cb_srtp {
                        match srtp_ctx.lock() {
                            Ok(mut ctx) => match ctx.protect_rtp(&packet) {
                                Ok(encrypted) => {
                                    // Log first SRTP protect
                                    static PROTECT_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                                    if !PROTECT_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                        log::info!("SRTP TX: first packet protected (plain len={}, encrypted len={})", packet.len(), encrypted.len());
                                    }
                                    encrypted
                                }
                                Err(e) => {
                                    log::error!("SRTP protect failed: {}", e);
                                    continue;
                                }
                            },
                            Err(_) => packet,
                        }
                    } else {
                        log::debug!("TX: No SRTP context, sending plain RTP");
                        packet
                    };

                    #[cfg(not(feature = "srtp"))]
                    let send_packet = packet;

                    cb_counters.record_sent(send_packet.len() as u64);

                    // Log first audio packet and then periodically (every 50 packets = ~1 second)
                    let count = cb_mic_count.fetch_add(1, Ordering::Relaxed);
                    if !cb_first_logged.swap(true, Ordering::Relaxed) {
                        log::info!("Audio TX: first mic packet captured, rms={:.4}", rms);
                    } else if count % 50 == 0 {
                        log::debug!(
                            "Audio TX: sent {} mic packets, seq={}, rms={:.4}",
                            count + 1,
                            current_seq,
                            rms
                        );
                    }

                    let dest = cb_learned.lock().ok().and_then(|g| *g).unwrap_or(remote);
                    let socket = cb_socket.clone();
                    cb_rt.spawn(async move {
                        let _ = socket.send_to(&send_packet, dest).await;
                    });
                }
            },
            |err| log::error!("Audio input error: {}", err),
            None,
        )
        .map_err(|e| Error::device(format!("Failed to build input stream: {}", e)))?;

    stream
        .play()
        .map_err(|e| Error::device(format!("Failed to start input: {}", e)))?;
    let device_name = device
        .description()
        .ok()
        .map_or_else(|| "unknown".to_string(), |d| d.name().to_string());
    log::info!(
        "Audio TX: microphone stream started on device '{}'",
        device_name
    );

    // Keepalive/silence packet sender - sends RTP every 20ms when muted or as initial NAT punch
    // This ensures NAT pinholes stay open and symmetric RTP works even without mic input
    let keepalive_running = running.clone();
    let keepalive_muted = muted.clone();
    let keepalive_socket = socket.clone();
    let keepalive_learned = learned_remote.clone();
    let keepalive_counters = counters.clone();
    let keepalive_encoder = encoder.clone();
    #[cfg(feature = "srtp")]
    let keepalive_srtp = _srtp.clone();

    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                log::error!("Failed to create keepalive runtime: {}", e);
                return;
            }
        };

        rt.block_on(async {
            // Send initial NAT punch packets immediately (5 packets over 100ms)
            for _ in 0..5 {
                if !keepalive_running.load(Ordering::Relaxed) {
                    return;
                }

                let current_seq = seq.fetch_add(1, Ordering::Relaxed);
                let current_ts = ts.fetch_add(samples_per_frame as u32, Ordering::Relaxed);

                let header = RtpHeader::new(pt, current_seq, current_ts, ssrc);
                let mut packet = header.to_bytes();

                if let Ok(mut enc) = keepalive_encoder.lock() {
                    enc.encode(&silence_pcm, &mut packet);
                }

                #[cfg(feature = "srtp")]
                let send_packet = if let Some(ref srtp_ctx) = keepalive_srtp {
                    match srtp_ctx.lock() {
                        Ok(mut ctx) => ctx.protect_rtp(&packet).unwrap_or(packet),
                        Err(_) => packet,
                    }
                } else {
                    packet
                };

                #[cfg(not(feature = "srtp"))]
                let send_packet = packet;

                keepalive_counters.record_sent(send_packet.len() as u64);

                let dest = keepalive_learned
                    .lock()
                    .ok()
                    .and_then(|g| *g)
                    .unwrap_or(remote);
                let _ = keepalive_socket.send_to(&send_packet, dest).await;

                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }

            log::info!("Sent initial NAT punch packets to {}", remote);

            // Continue sending keepalive silence packets when muted (every 20ms)
            while keepalive_running.load(Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;

                if !keepalive_running.load(Ordering::Relaxed) {
                    break;
                }

                // Only send keepalive when muted (normal audio TX handles unmuted case)
                if !keepalive_muted.load(Ordering::Relaxed) {
                    continue;
                }

                let current_seq = seq.fetch_add(1, Ordering::Relaxed);
                let current_ts = ts.fetch_add(samples_per_frame as u32, Ordering::Relaxed);

                let header = RtpHeader::new(pt, current_seq, current_ts, ssrc);
                let mut packet = header.to_bytes();

                if let Ok(mut enc) = keepalive_encoder.lock() {
                    enc.encode(&silence_pcm, &mut packet);
                }

                #[cfg(feature = "srtp")]
                let send_packet = if let Some(ref srtp_ctx) = keepalive_srtp {
                    match srtp_ctx.lock() {
                        Ok(mut ctx) => ctx.protect_rtp(&packet).unwrap_or(packet),
                        Err(_) => packet,
                    }
                } else {
                    packet
                };

                #[cfg(not(feature = "srtp"))]
                let send_packet = packet;

                keepalive_counters.record_sent(send_packet.len() as u64);

                let dest = keepalive_learned
                    .lock()
                    .ok()
                    .and_then(|g| *g)
                    .unwrap_or(remote);
                let _ = keepalive_socket.send_to(&send_packet, dest).await;
            }
        });
    });

    while running.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    drop(stream);
    drop(rt); // Shutdown the TX runtime after stream is done
    Ok(())
}

#[cfg(feature = "device")]
fn run_audio_rx(
    socket: Arc<UdpSocket>,
    running: Arc<AtomicBool>,
    mut decoder: Box<dyn AudioDecoder>,
    counters: RtpCounters,
    learned_remote: Arc<std::sync::Mutex<Option<SocketAddr>>>,
    _srtp: Option<SharedSrtp>,
    recorder_handle: Arc<std::sync::Mutex<Option<RecorderHandle>>>,
) -> Result<()> {
    use std::collections::VecDeque;

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| Error::device("No output device"))?;

    let default_config = device
        .default_output_config()
        .map_err(|e| Error::device(format!("No output config: {}", e)))?;

    let native_rate = default_config.sample_rate();
    log::info!("Audio RX: native rate = {} Hz", native_rate);

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: default_config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let codec_rate = 8000u32;

    // Pre-buffer: accumulate 60ms of audio before starting playback to absorb jitter
    let pre_buffer_samples = (native_rate as usize * 60) / 1000; // 60ms at native rate
    let max_buffer_samples = native_rate as usize / 2; // 500ms max buffer

    let sample_buffer: Arc<std::sync::Mutex<VecDeque<f32>>> = Arc::new(std::sync::Mutex::new(
        VecDeque::with_capacity(max_buffer_samples),
    ));
    let rx_buffer = sample_buffer.clone();
    let playback_started = Arc::new(AtomicBool::new(false));
    let pb_started = playback_started.clone();

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if let Ok(mut buffer) = rx_buffer.lock() {
                    // Don't start draining until we have enough pre-buffered audio
                    if !pb_started.load(Ordering::Relaxed) {
                        if buffer.len() >= pre_buffer_samples {
                            pb_started.store(true, Ordering::Relaxed);
                            log::info!(
                                "Audio RX: pre-buffer filled ({} samples), starting playback",
                                buffer.len()
                            );
                        } else {
                            // Still buffering — output silence
                            for out in data.iter_mut() {
                                *out = 0.0;
                            }
                            return;
                        }
                    }

                    // If buffer runs critically low, pause playback to re-buffer
                    // (avoids constant underrun clicks)
                    if buffer.is_empty() {
                        pb_started.store(false, Ordering::Relaxed);
                        log::debug!("Audio RX: buffer underrun, re-buffering");
                        for out in data.iter_mut() {
                            *out = 0.0;
                        }
                        return;
                    }

                    for out in data.iter_mut() {
                        *out = buffer.pop_front().unwrap_or(0.0);
                    }
                } else {
                    for out in data.iter_mut() {
                        *out = 0.0;
                    }
                }
            },
            |err| log::error!("Audio output error: {}", err),
            None,
        )
        .map_err(|e| Error::device(format!("Failed to build output stream: {}", e)))?;

    stream
        .play()
        .map_err(|e| Error::device(format!("Failed to start output: {}", e)))?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::device(format!("Failed to create runtime: {}", e)))?;

    rt.block_on(async {
        use crate::jitter::{JitterBuffer, JitterConfig, JitterMode};

        let mut buf = [0u8; 2048];
        let mut last_transit: Option<i64> = None;
        let mut first_seq: Option<u16> = None;

        // Jitter buffer: 40ms target delay, adapts between 20-150ms
        let jitter_config = JitterConfig {
            mode: JitterMode::Adaptive {
                target_ms: 40,
                min_ms: 20,
                max_ms: 150,
            },
            clock_rate: codec_rate,
            max_packets: 50,
        };
        let mut jitter_buf = JitterBuffer::new(jitter_config);

        // Playout interval: pop from jitter buffer every 20ms
        let mut playout_interval = tokio::time::interval(std::time::Duration::from_millis(20));
        playout_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Persistent resampler: maintains state across frames for glitch-free audio
        // 160 samples = 20ms at 8kHz codec rate
        let mut rx_resampler = StreamResampler::new(codec_rate, native_rate, 160);

        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }

            tokio::select! {
                // Receive RTP packets and push into jitter buffer
                recv = socket.recv_from(&mut buf) => {
                    match recv {
                        Ok((len, from_addr)) => {
                            // Learn remote address for symmetric RTP
                            if let Ok(mut lr) = learned_remote.lock()
                                && lr.is_none()
                            {
                                log::info!("Comedia: learned remote RTP address {}", from_addr);
                                *lr = Some(from_addr);
                            }

                            #[cfg(feature = "srtp")]
                            let rtp_data: Vec<u8> = if let Some(ref srtp_ctx) = _srtp {
                                match srtp_ctx.lock() {
                                    Ok(mut ctx) => match ctx.unprotect_rtp(&buf[..len]) {
                                        Ok(decrypted) => {
                                            static UNPROTECT_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                                            let count = UNPROTECT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                            if count == 0 {
                                                log::info!("SRTP RX: first packet unprotected successfully (encrypted len={}, decrypted len={})", len, decrypted.len());
                                            }
                                            decrypted
                                        }
                                        Err(e) => {
                                            log::warn!("SRTP unprotect failed: {}", e);
                                            continue;
                                        }
                                    },
                                    Err(_) => buf[..len].to_vec(),
                                }
                            } else {
                                buf[..len].to_vec()
                            };

                            #[cfg(not(feature = "srtp"))]
                            let rtp_data: Vec<u8> = buf[..len].to_vec();

                            // Track stats
                            let seq = parse_sequence(&rtp_data);
                            let rtp_ts = parse_timestamp(&rtp_data);

                            if let Some(s) = seq {
                                if first_seq.is_none() {
                                    first_seq = Some(s);
                                }
                                counters.record_received(len as u64, s);
                            }

                            // Jitter calculation
                            if let Some(ts) = rtp_ts {
                                let arrival = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_micros() as i64;
                                let transit = arrival - (ts as i64 * 125);
                                if let Some(prev) = last_transit {
                                    let d = (transit - prev).unsigned_abs();
                                    counters.update_jitter(d);
                                }
                                last_transit = Some(transit);
                            }

                            // Push into jitter buffer instead of decoding immediately
                            if let Some((header, payload)) = parse_rtp(&rtp_data) {
                                let s = header.sequence;
                                let t = header.timestamp;
                                jitter_buf.push(s, t, payload.to_vec());
                            }
                        }
                        Err(e) => {
                            log::error!("RTP recv error: {}", e);
                        }
                    }
                }

                // Playout timer: pop from jitter buffer every 20ms
                _ = playout_interval.tick() => {
                    if let Some(pkt) = jitter_buf.pop() {
                        // Decode the payload (empty payload = loss concealment)
                        let mut pcm = Vec::with_capacity(160);
                        if pkt.payload.is_empty() {
                            // Packet loss concealment: repeat last frame as silence
                            pcm.resize(160, 0i16);
                        } else {
                            decoder.decode(&pkt.payload, &mut pcm);
                        }

                        // Record RX audio if recording is active
                        if let Ok(handle) = recorder_handle.lock() {
                            if let Some(ref rec) = *handle {
                                rec.add_samples(&pcm);
                            }
                        }

                        let f32_samples = i16_to_f32(&pcm);
                        let resampled = rx_resampler.process(&f32_samples);

                        if let Ok(mut buffer) = sample_buffer.lock() {
                            if buffer.len() + resampled.len() > max_buffer_samples {
                                let excess = buffer.len() + resampled.len() - max_buffer_samples;
                                buffer.drain(..excess);
                                log::debug!("Audio RX: buffer overflow, drained {} samples", excess);
                            }
                            buffer.extend(resampled.iter());
                        }
                    }
                }
            }
        }
    });

    drop(stream);
    Ok(())
}

async fn run_rtcp(
    socket: Arc<UdpSocket>,
    remote_addr: SocketAddr,
    ssrc: u32,
    running: Arc<AtomicBool>,
    counters: RtpCounters,
    _tx_srtp: Option<SharedSrtp>,
    _rx_srtp: Option<SharedSrtp>,
) {
    let mut remote_ssrc: u32 = 0;
    let mut buf = [0u8; 512];

    while running.load(Ordering::Relaxed) {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        if !running.load(Ordering::Relaxed) {
            break;
        }

        // Send Sender Report
        let stats = counters.snapshot();
        let sr = build_rtcp_sr(ssrc, stats.packets_sent as u32, stats.bytes_sent as u32);

        #[cfg(feature = "srtp")]
        let sr_to_send = if let Some(ref srtp_ctx) = _tx_srtp {
            match srtp_ctx.lock() {
                Ok(mut ctx) => ctx.protect_rtcp(&sr).unwrap_or(sr),
                Err(_) => sr,
            }
        } else {
            sr
        };

        #[cfg(not(feature = "srtp"))]
        let sr_to_send = sr;

        let _ = socket.send_to(&sr_to_send, remote_addr).await;

        // Send Receiver Report if we know remote SSRC
        if remote_ssrc != 0 {
            let received = stats.packets_received;
            let expected = counters.expected_packets.load(Ordering::Relaxed);
            let lost = expected.saturating_sub(received);
            let loss_fraction = if expected > 0 {
                ((lost * 256) / expected) as u8
            } else {
                0
            };
            let rr = build_rtcp_rr(
                ssrc,
                remote_ssrc,
                loss_fraction,
                lost as u32,
                counters.highest_seq.load(Ordering::Relaxed),
                (counters.jitter_us.load(Ordering::Relaxed) / 125) as u32,
            );

            #[cfg(feature = "srtp")]
            let rr_to_send = if let Some(ref srtp_ctx) = _tx_srtp {
                match srtp_ctx.lock() {
                    Ok(mut ctx) => ctx.protect_rtcp(&rr).unwrap_or(rr),
                    Err(_) => rr,
                }
            } else {
                rr
            };

            #[cfg(not(feature = "srtp"))]
            let rr_to_send = rr;

            let _ = socket.send_to(&rr_to_send, remote_addr).await;
        }

        // Receive RTCP
        if let Ok(Ok((len, _))) = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            socket.recv_from(&mut buf),
        )
        .await
        {
            #[cfg(feature = "srtp")]
            let rtcp_data: Vec<u8> = if let Some(ref srtp_ctx) = _rx_srtp {
                match srtp_ctx.lock() {
                    Ok(mut ctx) => ctx
                        .unprotect_rtcp(&buf[..len])
                        .unwrap_or_else(|_| buf[..len].to_vec()),
                    Err(_) => buf[..len].to_vec(),
                }
            } else {
                buf[..len].to_vec()
            };

            #[cfg(not(feature = "srtp"))]
            let rtcp_data: Vec<u8> = buf[..len].to_vec();

            if rtcp_data.len() >= 8 && (rtcp_data[1] == 200 || rtcp_data[1] == 201) {
                remote_ssrc =
                    u32::from_be_bytes([rtcp_data[4], rtcp_data[5], rtcp_data[6], rtcp_data[7]]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_codec_type_properties() {
        // Test that codec type constants are correct
        assert_eq!(CodecType::Pcmu.payload_type(), 0);
        assert_eq!(CodecType::Pcma.payload_type(), 8);
        assert_eq!(CodecType::Pcmu.clock_rate(), 8000);
        assert_eq!(CodecType::Pcmu.samples_per_frame(), 160);
    }

    #[tokio::test]
    async fn test_media_session_start_invalid_port() {
        // Try to bind to a privileged port (requires root)
        let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5000);
        let result = MediaSession::start(80, remote, CodecType::Pcmu).await;

        // Should fail on non-root systems
        // This tests error handling path
        if result.is_err() {
            assert!(matches!(result, Err(Error::Network(_))));
        }
    }

    #[tokio::test]
    async fn test_media_session_basic_creation() {
        // Use a random high port to avoid conflicts
        let port = 50000 + (rand::random::<u16>() % 10000);
        let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5000);

        // This will fail in CI without audio devices, but tests the creation path
        let result = MediaSession::start(port, remote, CodecType::Pcmu).await;

        // In environments without audio, this fails at device setup
        // In environments with audio, it succeeds
        // Either way, we're testing the code path
        match result {
            Ok(session) => {
                // Session created successfully
                assert!(!session.is_muted());
                session.stop();
            }
            Err(e) => {
                // Expected on CI without audio devices
                assert!(
                    matches!(e, Error::Device(_)) || matches!(e, Error::Network(_)),
                    "Unexpected error type: {:?}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_rtp_counters_initialization() {
        let counters = RtpCounters::new("PCMU");
        let stats = counters.snapshot();

        assert_eq!(stats.packets_sent, 0);
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.packets_received, 0);
        assert_eq!(stats.packets_lost, 0);
    }

    #[test]
    fn test_socket_addr_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)), 5060);
        assert_eq!(addr.port(), 5060);
        assert_eq!(addr.ip(), IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)));
    }

    #[test]
    fn test_create_encoder_decoder() {
        // Test encoder creation
        let encoder = create_encoder(CodecType::Pcmu);
        assert!(encoder.is_ok());

        let encoder = create_encoder(CodecType::Pcma);
        assert!(encoder.is_ok());

        // Test decoder creation
        let decoder = create_decoder(CodecType::Pcmu);
        assert!(decoder.is_ok());

        let decoder = create_decoder(CodecType::Pcma);
        assert!(decoder.is_ok());
    }

    #[cfg(feature = "srtp")]
    #[test]
    fn test_srtp_context_for_session() {
        use crate::srtp::SrtpContext;

        let (_ctx, key) = SrtpContext::generate().unwrap();
        assert!(!key.is_empty());

        // Context should be able to protect/unprotect
        let mut test_rtp = vec![
            0x80, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xA0, 0x12, 0x34, 0x56, 0x78,
        ];
        test_rtp.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let mut ctx_clone = SrtpContext::from_base64(&key).unwrap();
        let protected = ctx_clone.protect_rtp(&test_rtp);
        assert!(protected.is_ok());
    }
}

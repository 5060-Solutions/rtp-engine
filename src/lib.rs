//! # rtp-engine
//!
//! A pure Rust RTP media engine for VoIP applications.
//!
//! This crate provides the building blocks for real-time audio communication:
//!
//! - **Codecs**: G.711 (μ-law/A-law) and Opus encoding/decoding
//! - **RTP/RTCP**: Packet construction, parsing, and statistics
//! - **SRTP**: Secure RTP with AES-CM-128-HMAC-SHA1-80
//! - **Audio devices**: Cross-platform capture and playback via cpal
//! - **Resampling**: Sample rate conversion between codecs and devices
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rtp_engine::{MediaSession, CodecType};
//! use std::net::SocketAddr;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let remote: SocketAddr = "192.168.1.100:5004".parse()?;
//!     
//!     // Start a media session with G.711 μ-law
//!     let session = MediaSession::start(10000, remote, CodecType::Pcmu).await?;
//!     
//!     // The session captures from mic and plays to speaker automatically
//!     // Send DTMF
//!     session.send_dtmf("1");
//!     
//!     // Mute/unmute
//!     session.set_mute(true);
//!     
//!     // Get statistics
//!     let stats = session.stats();
//!     println!("Packets sent: {}", stats.packets_sent);
//!     
//!     // Stop when done
//!     session.stop();
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `g711` (default): G.711 μ-law and A-law codecs
//! - `opus`: Opus codec support (requires libopus)
//! - `srtp`: SRTP/SRTCP encryption
//! - `device`: Audio device capture/playback via cpal

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod codec;
pub mod error;
pub mod jitter;
pub mod rtp;
pub mod stun;

#[cfg(feature = "srtp")]
pub mod srtp;

#[cfg(feature = "device")]
pub mod device;

#[cfg(feature = "device")]
pub use device::{
    AudioDevice, AudioDevices, list_all_devices, list_input_devices, list_output_devices,
};

pub mod recorder;
pub mod resample;
mod session;

// Re-exports for convenience
pub use codec::CodecType;
pub use error::{Error, Result};
pub use jitter::{JitterBuffer, JitterConfig, JitterMode, JitterStats};
pub use recorder::{CallRecorder, RecorderHandle, generate_recording_filename};
pub use resample::{f32_to_i16, i16_to_f32, resample_linear, resample_linear_i16};
pub use rtp::RtpStats;
pub use session::MediaSession;
pub use stun::{
    DEFAULT_STUN_SERVERS, StunResult, discover as stun_discover, discover_public_address,
};

#[cfg(feature = "srtp")]
pub use srtp::SrtpContext;

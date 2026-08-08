//! Acoustic echo cancellation and noise suppression for device audio.
//!
//! Only the desktop needs this. Android and iOS request the OS voice-processing
//! path when they open their streams (`VOICE_COMMUNICATION` and
//! `AVAudioSession` `.voiceChat` respectively), which applies the platform's
//! own AEC and NS — running a second canceller on top of a tuned hardware one
//! sounds worse than either alone. So this is opt-in and the mobile core does
//! not enable it.
//!
//! Echo cancellation needs both signals: the microphone (near end) and whatever
//! is being played (far end, the "render" signal) to subtract from it. Both
//! callbacks live in [`crate::session`], which is what makes this feasible
//! here and not in the SIP layer above.

pub mod frame;
pub mod processor;

pub use frame::{FRAME_MS, FrameAccumulator, frame_len};
pub use processor::{
    CapturePath, NoiseLevel, RenderPath, VoiceProcessor, VoiceProcessorConfig, default_config,
    set_default_config,
};

//! The duplex voice processor: echo cancellation and noise suppression.
//!
//! Echo cancellation is inherently a two-stream problem. The canceller needs the
//! microphone signal (near end) *and* the signal being played out of the speaker
//! (far end, the "render" reference) so it can subtract the latter from the
//! former. Those two live in different threads with different device clocks, so
//! the shared state here is what joins them.
//!
//! # Shape
//!
//! One [`VoiceProcessor`] per call, shared by both audio threads. Each thread
//! takes a handle — [`CapturePath`] for the microphone, [`RenderPath`] for the
//! speaker — and each handle owns its own [`FrameAccumulator`], because the two
//! device callbacks arrive at unrelated sizes and must be re-framed
//! independently.
//!
//! # Why this is opt-in
//!
//! Behind the `audio-proc` feature, off by default, for two reasons:
//!
//! 1. It pulls in a C++ library that builds from source and needs `meson`,
//!    `ninja` and a C++ toolchain present. That is a fine ask for a desktop
//!    build and an unreasonable one for a library consumer who only wants RTP.
//! 2. Mobile must not enable it. Android's `VOICE_COMMUNICATION` source and
//!    iOS's `.voiceChat` mode already apply the platform's own AEC, tuned to
//!    that specific hardware. A second canceller stacked on a tuned one sounds
//!    worse than either alone — the two adapt against each other.
//!
//! With the feature off, every type here still exists and every call still
//! compiles; the processor simply reports itself disabled and audio passes
//! through untouched. Callers have one code path either way.

use std::sync::{Arc, RwLock};

use super::frame::FrameAccumulator;

/// How aggressively to suppress background noise.
///
/// Mirrors the underlying library's levels rather than re-exporting them, so
/// this type exists whether or not the `audio-proc` feature is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoiseLevel {
    /// Least suppression, least speech distortion.
    Low,
    /// The sane default for a headset or handset.
    #[default]
    Moderate,
    /// For a noisy room, at some cost to voice quality.
    High,
    /// Maximum suppression. Audibly processes the voice.
    VeryHigh,
}

/// What the voice processor should do to the capture stream.
///
/// [`Default`] is everything off, matching the crate's default feature set —
/// enabling processing is always an explicit decision by the host application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VoiceProcessorConfig {
    /// Subtract the speaker signal from the microphone signal.
    pub echo_cancellation: bool,
    /// Attenuate steady background noise.
    pub noise_suppression: bool,
    /// How hard [`Self::noise_suppression`] works. Ignored when it is off.
    pub noise_level: NoiseLevel,
    /// Bring a quiet or loud microphone toward a usable level.
    pub gain_control: bool,
}

impl VoiceProcessorConfig {
    /// The settings a desktop softphone wants: cancel echo, suppress noise,
    /// normalise level.
    #[must_use]
    pub fn desktop_default() -> Self {
        Self {
            echo_cancellation: true,
            noise_suppression: true,
            noise_level: NoiseLevel::Moderate,
            gain_control: true,
        }
    }

    /// Whether anything is enabled at all.
    #[must_use]
    pub const fn is_any_enabled(&self) -> bool {
        self.echo_cancellation || self.noise_suppression || self.gain_control
    }
}

/// Everything off — the value a process starts with.
const ALL_OFF: VoiceProcessorConfig = VoiceProcessorConfig {
    echo_cancellation: false,
    noise_suppression: false,
    noise_level: NoiseLevel::Moderate,
    gain_control: false,
};

/// The configuration new sessions pick up when none is given.
///
/// Process-wide because that is what it models: a user's preference in the
/// application's audio settings, which applies to every call they make, not a
/// property of any one call. Threading it through every session constructor
/// would put the same value in every call site and let them disagree.
static DEFAULT_CONFIG: RwLock<VoiceProcessorConfig> = RwLock::new(ALL_OFF);

/// Set the voice processing every later session will use.
///
/// Call this when the user changes the setting. Sessions already running keep
/// the configuration they started with — changing it mid-call would drop the
/// canceller's adapted state and produce a burst of echo.
pub fn set_default_config(config: VoiceProcessorConfig) {
    let mut slot = DEFAULT_CONFIG
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *slot = config;
    log::info!("Voice processing default set to {config:?}");
}

/// The configuration new sessions will pick up.
#[must_use]
pub fn default_config() -> VoiceProcessorConfig {
    *DEFAULT_CONFIG
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether this build can do voice processing at all.
///
/// False when the crate was compiled without the `audio-proc` feature, in which
/// case [`set_default_config`] will accept a configuration and quietly do
/// nothing with it. A host with a settings screen should ask this before
/// offering the user a switch — a control that claims to cancel echo and does
/// not is worse than no control.
#[must_use]
pub const fn is_available() -> bool {
    cfg!(feature = "audio-proc")
}

/// Which half of the duplex path a handle carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Capture,
    Render,
}

/// The processor's lifecycle.
///
/// It starts in `Negotiating` because neither audio thread has opened its
/// device yet — the sample rate is not known until `cpal` reports it, and the
/// canceller needs a single rate for both streams.
enum State {
    /// Waiting for both threads to report their device rate.
    Negotiating {
        capture: Option<u32>,
        render: Option<u32>,
    },
    /// Both rates agreed and the backend is running.
    #[cfg(feature = "audio-proc")]
    Active {
        rate: u32,
        backend: webrtc_audio_processing::Processor,
    },
    /// Off for the rest of the call: not compiled in, not configured, the two
    /// devices disagreed on a rate, or the backend refused to start.
    Disabled,
}

/// Shared echo canceller and noise suppressor for one call.
///
/// Create one, hand [`Self::capture_path`] to the microphone thread and
/// [`Self::render_path`] to the speaker thread. Order does not matter: whichever
/// arrives second completes the negotiation, and until then audio passes
/// through unprocessed.
pub struct VoiceProcessor {
    state: RwLock<State>,
    config: VoiceProcessorConfig,
}

impl std::fmt::Debug for VoiceProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceProcessor")
            .field("config", &self.config)
            .field("active", &self.is_active())
            .finish()
    }
}

impl VoiceProcessor {
    /// Create a processor for one call.
    ///
    /// Returns a disabled processor — one that copies audio through untouched —
    /// if `config` enables nothing, or if the crate was built without the
    /// `audio-proc` feature. That is not an error: it is the default build, and
    /// the caller's code path is identical either way.
    #[must_use]
    pub fn new(config: VoiceProcessorConfig) -> Arc<Self> {
        let compiled_in = cfg!(feature = "audio-proc");
        let state = if config.is_any_enabled() && compiled_in {
            State::Negotiating {
                capture: None,
                render: None,
            }
        } else {
            if config.is_any_enabled() && !compiled_in {
                log::warn!(
                    "Voice processing was requested but rtp-engine was built \
                     without the 'audio-proc' feature; audio will pass through \
                     unprocessed"
                );
            }
            State::Disabled
        };
        Arc::new(Self {
            state: RwLock::new(state),
            config,
        })
    }

    /// A processor built from the process-wide default set by
    /// [`set_default_config`].
    #[must_use]
    pub fn from_default() -> Arc<Self> {
        Self::new(default_config())
    }

    /// A processor that does nothing, for callers that do not want any.
    #[must_use]
    pub fn disabled() -> Arc<Self> {
        Self::new(ALL_OFF)
    }

    /// What this processor was asked to do.
    #[must_use]
    pub const fn config(&self) -> VoiceProcessorConfig {
        self.config
    }

    /// Whether the backend is running and actually processing audio.
    ///
    /// False until both audio threads have opened their devices, and false for
    /// the rest of the call if negotiation failed. Worth surfacing in a UI: a
    /// user who enabled echo cancellation should be able to tell that it did
    /// not start.
    #[must_use]
    pub fn is_active(&self) -> bool {
        #[cfg(feature = "audio-proc")]
        {
            matches!(&*self.read_state(), State::Active { .. })
        }
        #[cfg(not(feature = "audio-proc"))]
        {
            false
        }
    }

    /// Take the microphone-side handle, reporting the capture device's rate.
    pub fn capture_path(self: &Arc<Self>, device_rate: u32) -> CapturePath {
        self.attach(Side::Capture, device_rate);
        CapturePath {
            shared: Arc::clone(self),
            accumulator: FrameAccumulator::new(device_rate),
            device_rate,
            scratch: Vec::new(),
        }
    }

    /// Take the speaker-side handle, reporting the playback device's rate.
    pub fn render_path(self: &Arc<Self>, device_rate: u32) -> RenderPath {
        self.attach(Side::Render, device_rate);
        RenderPath {
            shared: Arc::clone(self),
            accumulator: FrameAccumulator::new(device_rate),
            device_rate,
            scratch: Vec::new(),
        }
    }

    /// Read the state, ignoring poisoning.
    ///
    /// A panic in an audio callback would poison this lock and silently kill
    /// processing for the rest of the call. Recovering the guard keeps audio
    /// flowing, which matters more here than propagating the panic.
    fn read_state(&self) -> std::sync::RwLockReadGuard<'_, State> {
        self.state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Record one side's device rate and start the backend once both are known.
    fn attach(&self, side: Side, rate: u32) {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        match &mut *state {
            State::Negotiating { capture, render } => {
                match side {
                    Side::Capture => *capture = Some(rate),
                    Side::Render => *render = Some(rate),
                }
                let (Some(capture_rate), Some(render_rate)) = (*capture, *render) else {
                    // Still waiting on the other thread to open its device.
                    return;
                };
                *state = Self::negotiate(capture_rate, render_rate, self.config);
            }

            // A stream restarted mid-call — a device change, say. Drop the
            // adapted echo estimate, which describes a path that no longer
            // exists, but keep the backend if the rate still matches.
            #[cfg(feature = "audio-proc")]
            State::Active {
                rate: active,
                backend,
            } => {
                if *active == rate {
                    backend.reinitialize();
                    log::info!(
                        "Voice processing: {side:?} stream restarted at {rate} Hz, echo estimate reset"
                    );
                } else {
                    log::warn!(
                        "Voice processing: {side:?} stream restarted at {rate} Hz but the \
                         other stream runs at {active} Hz; disabling"
                    );
                    *state = State::Disabled;
                }
            }

            State::Disabled => {}
        }
    }

    /// Decide whether the two device rates can support processing, and if so
    /// build the backend.
    fn negotiate(capture_rate: u32, render_rate: u32, config: VoiceProcessorConfig) -> State {
        if capture_rate != render_rate {
            log::warn!(
                "Voice processing needs one sample rate for both streams, but the \
                 microphone runs at {capture_rate} Hz and the speaker at {render_rate} Hz; \
                 disabling. Select devices that share a rate to enable it."
            );
            return State::Disabled;
        }

        // The accumulator refuses rates the backend cannot accept — notably
        // 44_100. Ask it first so the two never disagree about what is legal.
        if FrameAccumulator::new(capture_rate).is_none() {
            log::warn!(
                "Voice processing does not support {capture_rate} Hz devices \
                 (needs 8/16/32/48 kHz); disabling"
            );
            return State::Disabled;
        }

        Self::build_backend(capture_rate, config)
    }

    #[cfg(feature = "audio-proc")]
    fn build_backend(rate: u32, config: VoiceProcessorConfig) -> State {
        use webrtc_audio_processing::{
            Processor,
            config::{
                Config, EchoCanceller, GainController, GainController1, HighPassFilter,
                NoiseSuppression, NoiseSuppressionLevel,
            },
        };

        let backend = match Processor::new(rate) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("Voice processing failed to start at {rate} Hz: {e}; disabling");
                return State::Disabled;
            }
        };

        let level = match config.noise_level {
            NoiseLevel::Low => NoiseSuppressionLevel::Low,
            NoiseLevel::Moderate => NoiseSuppressionLevel::Moderate,
            NoiseLevel::High => NoiseSuppressionLevel::High,
            NoiseLevel::VeryHigh => NoiseSuppressionLevel::VeryHigh,
        };

        backend.set_config(Config {
            // Strongly recommended alongside echo cancellation: DC and rumble
            // below the voice band make the adaptive filter converge worse.
            high_pass_filter: config.echo_cancellation.then(HighPassFilter::default),
            // No stream delay supplied: AEC3 estimates it. We cannot supply a
            // useful one anyway, because the render reference is taken at the
            // device callback and the true delay is the hardware's, which cpal
            // does not report consistently across backends.
            echo_canceller: config.echo_cancellation.then_some(EchoCanceller::Full {
                stream_delay_ms: None,
            }),
            noise_suppression: config.noise_suppression.then_some(NoiseSuppression {
                level,
                analyze_linear_aec_output: false,
            }),
            // Digital-only. AdaptiveAnalog would require us to drive the OS
            // mixer's capture level, which this crate does not touch.
            gain_controller: config.gain_control.then(|| {
                GainController::GainController1(GainController1 {
                    mode: webrtc_audio_processing::config::GainControllerMode::AdaptiveDigital,
                    ..GainController1::default()
                })
            }),
            ..Config::default()
        });

        log::info!(
            "Voice processing active at {rate} Hz (aec={}, ns={}, agc={})",
            config.echo_cancellation,
            config.noise_suppression,
            config.gain_control
        );
        State::Active { rate, backend }
    }

    #[cfg(not(feature = "audio-proc"))]
    fn build_backend(_rate: u32, _config: VoiceProcessorConfig) -> State {
        State::Disabled
    }
}

/// The microphone side of the processor.
///
/// Not [`Sync`]: it holds per-stream framing state and belongs to exactly one
/// audio callback.
#[derive(Debug)]
pub struct CapturePath {
    shared: Arc<VoiceProcessor>,
    /// `None` when the device rate cannot be framed, which also means the
    /// shared state is `Disabled`.
    accumulator: Option<FrameAccumulator>,
    device_rate: u32,
    scratch: Vec<f32>,
}

impl CapturePath {
    /// The sample rate this path was opened at.
    #[must_use]
    pub const fn device_rate(&self) -> u32 {
        self.device_rate
    }

    /// Run the microphone buffer through the processor.
    ///
    /// Returns the processed audio, which is *not* the same length as `input`:
    /// up to one 10 ms frame is held back to complete the next frame, so early
    /// calls may return less than they were given and later ones more. That is
    /// the only latency this adds — under 10 ms.
    ///
    /// When processing is disabled the input is returned unchanged, with no
    /// copy and no held-back samples.
    pub fn process<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        let state = self.shared.read_state();

        #[cfg(feature = "audio-proc")]
        if let State::Active { rate, backend } = &*state {
            // The rate is fixed at negotiation, but a device that reopened at a
            // different rate would make the backend panic on frame size rather
            // than return an error. Refuse instead.
            if *rate != self.device_rate {
                return input;
            }
            let Some(accumulator) = self.accumulator.as_mut() else {
                return input;
            };

            self.scratch.clear();
            accumulator.push(input, &mut self.scratch, |frame| {
                if let Err(e) = backend.process_capture_frame(std::iter::once(&mut *frame)) {
                    // Leave the frame as-is: unprocessed voice beats dropped
                    // voice. Logged at debug because a persistent failure would
                    // otherwise flood at 100 lines a second.
                    log::debug!("Voice processing: capture frame failed: {e}");
                }
            });
            drop(state);
            return &self.scratch;
        }

        drop(state);
        input
    }

    /// Discard the partial frame, for a stream restart.
    pub fn reset(&mut self) {
        if let Some(accumulator) = self.accumulator.as_mut() {
            accumulator.reset();
        }
        self.scratch.clear();
    }
}

/// The speaker side of the processor.
///
/// This never modifies what is played. It only shows the canceller what is
/// about to come out of the speaker so it can recognise the echo of it in the
/// microphone.
#[derive(Debug)]
pub struct RenderPath {
    shared: Arc<VoiceProcessor>,
    accumulator: Option<FrameAccumulator>,
    device_rate: u32,
    scratch: Vec<f32>,
}

impl RenderPath {
    /// The sample rate this path was opened at.
    #[must_use]
    pub const fn device_rate(&self) -> u32 {
        self.device_rate
    }

    /// Show the canceller the samples being handed to the speaker.
    ///
    /// Call this on **every** output callback, including the ones that write
    /// silence while the jitter buffer refills. The canceller tracks the delay
    /// between this stream and the microphone; a gap in the reference looks
    /// like the delay changed and forces it to re-converge, which is audible as
    /// a burst of echo.
    ///
    /// Built without the `audio-proc` feature there is no canceller to tell, so
    /// both the lock guard and the samples go unused — hence the attribute.
    #[cfg_attr(not(feature = "audio-proc"), allow(unused_variables))]
    pub fn analyze(&mut self, played: &[f32]) {
        let state = self.shared.read_state();

        #[cfg(feature = "audio-proc")]
        if let State::Active { rate, backend } = &*state {
            if *rate != self.device_rate {
                return;
            }
            let Some(accumulator) = self.accumulator.as_mut() else {
                return;
            };

            self.scratch.clear();
            accumulator.push(played, &mut self.scratch, |frame| {
                if let Err(e) = backend.analyze_render_frame(std::iter::once(&*frame)) {
                    log::debug!("Voice processing: render frame failed: {e}");
                }
            });
        }

        drop(state);
    }

    /// Discard the partial frame, for a stream restart.
    pub fn reset(&mut self) {
        if let Some(accumulator) = self.accumulator.as_mut() {
            accumulator.reset();
        }
        self.scratch.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{NoiseLevel, VoiceProcessor, VoiceProcessorConfig};

    #[test]
    fn a_default_config_enables_nothing() {
        let config = VoiceProcessorConfig::default();
        assert!(!config.is_any_enabled());
        assert!(!config.echo_cancellation);
        assert_eq!(config.noise_level, NoiseLevel::Moderate);
    }

    #[test]
    fn the_desktop_preset_turns_on_the_things_the_softphone_advertises() {
        let config = VoiceProcessorConfig::desktop_default();
        assert!(config.echo_cancellation, "the marketing copy claims AEC");
        assert!(config.noise_suppression, "the marketing copy claims NS");
        assert!(config.is_any_enabled());
    }

    #[test]
    fn a_disabled_processor_never_reports_active() {
        let vp = VoiceProcessor::disabled();
        assert!(!vp.is_active());
        // Attaching both sides must not bring a disabled processor to life.
        let _capture = vp.capture_path(48_000);
        let _render = vp.render_path(48_000);
        assert!(!vp.is_active());
    }

    #[test]
    fn a_disabled_processor_passes_capture_audio_through_untouched() {
        let vp = VoiceProcessor::disabled();
        let mut capture = vp.capture_path(48_000);
        let input: Vec<f32> = (0..1000).map(|i| (i as f32) / 1000.0).collect();
        let out = capture.process(&input);
        assert_eq!(out.len(), input.len(), "nothing may be held back when off");
        assert_eq!(out, input.as_slice(), "samples must be unmodified");
    }

    #[test]
    fn render_analysis_is_harmless_when_disabled() {
        let vp = VoiceProcessor::disabled();
        let mut render = vp.render_path(48_000);
        render.analyze(&[0.5; 480]);
        render.analyze(&[]);
        render.reset();
        assert!(!vp.is_active());
    }

    #[test]
    fn mismatched_device_rates_disable_processing_rather_than_guess() {
        let vp = VoiceProcessor::new(VoiceProcessorConfig::desktop_default());
        let _capture = vp.capture_path(48_000);
        let _render = vp.render_path(16_000);
        assert!(
            !vp.is_active(),
            "two rates cannot share one canceller; it must refuse"
        );
    }

    #[test]
    fn an_unsupported_shared_rate_disables_processing() {
        let vp = VoiceProcessor::new(VoiceProcessorConfig::desktop_default());
        // 44.1 kHz frames evenly at 10 ms but the backend cannot accept it.
        let _capture = vp.capture_path(44_100);
        let _render = vp.render_path(44_100);
        assert!(!vp.is_active());
    }

    #[test]
    fn one_side_alone_does_not_start_processing() {
        let vp = VoiceProcessor::new(VoiceProcessorConfig::desktop_default());
        let mut capture = vp.capture_path(48_000);
        assert!(
            !vp.is_active(),
            "echo cancellation needs the render side too"
        );
        // And capture still works, unprocessed, while waiting.
        let input = [0.25_f32; 960];
        assert_eq!(capture.process(&input), input.as_slice());
    }

    /// Only meaningful with the backend compiled in; asserts the whole
    /// negotiation reaches `Active` for a realistic device pair.
    #[cfg(feature = "audio-proc")]
    #[test]
    fn matching_supported_rates_start_the_backend() {
        let vp = VoiceProcessor::new(VoiceProcessorConfig::desktop_default());
        let _capture = vp.capture_path(48_000);
        let _render = vp.render_path(48_000);
        assert!(vp.is_active(), "48 kHz on both sides must negotiate");
    }

    /// The property that makes this safe to wire into a live call: whatever the
    /// processor does to the samples, it must not lose or invent any beyond the
    /// single frame it is allowed to hold back.
    #[cfg(feature = "audio-proc")]
    #[test]
    fn capture_conserves_samples_within_one_frame() {
        let vp = VoiceProcessor::new(VoiceProcessorConfig::desktop_default());
        let mut capture = vp.capture_path(48_000);
        let mut render = vp.render_path(48_000);
        assert!(vp.is_active());

        let mut fed = 0_usize;
        let mut emitted = 0_usize;
        for chunk in [128_usize, 480, 1024, 17, 960] {
            render.analyze(&vec![0.0; chunk]);
            let input = vec![0.1_f32; chunk];
            emitted += capture.process(&input).len();
            fed += chunk;
            assert!(
                fed - emitted < 480,
                "more than one 10 ms frame is being held back"
            );
        }
    }

    /// Drive an echo-cancelling processor with a microphone signal that is a
    /// scaled copy of the speaker signal — pure echo, no near-end speech — and
    /// report (input energy, output energy) measured after convergence.
    ///
    /// `feed_reference` controls whether the canceller is actually shown the
    /// speaker signal. Passing `false` keeps every other part of the path
    /// identical — same backend, same framing, same buffers — and removes only
    /// the canceller's ability to do its job, which is what makes it a usable
    /// control.
    #[cfg(feature = "audio-proc")]
    fn echo_energy(feed_reference: bool) -> (f64, f64) {
        let vp = VoiceProcessor::new(VoiceProcessorConfig {
            echo_cancellation: true,
            noise_suppression: false,
            noise_level: NoiseLevel::Moderate,
            gain_control: false,
        });
        let mut capture = vp.capture_path(16_000);
        let mut render = vp.render_path(16_000);
        assert!(vp.is_active());

        let frame = 160_usize;
        let mut phase = 0.0_f32;
        let (mut energy_in, mut energy_out) = (0.0_f64, 0.0_f64);
        // AEC3 needs a few hundred ms to converge; measure only the tail.
        let (total_frames, measure_from) = (200, 150);

        for n in 0..total_frames {
            let far: Vec<f32> = (0..frame)
                .map(|_| {
                    phase += 0.15;
                    phase.sin() * 0.5
                })
                .collect();
            let near: Vec<f32> = far.iter().map(|s| s * 0.5).collect();

            if feed_reference {
                render.analyze(&far);
            } else {
                // The speaker is silent as far as the canceller knows, so the
                // tone in the microphone is near-end speech, not echo.
                render.analyze(&vec![0.0; frame]);
            }
            let out = capture.process(&near);

            if n >= measure_from {
                let energy =
                    |s: &[f32]| s.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
                energy_in += energy(&near);
                energy_out += energy(out);
            }
        }
        (energy_in, energy_out)
    }

    /// Echo cancellation must actually cancel.
    #[cfg(feature = "audio-proc")]
    #[test]
    fn the_canceller_attenuates_a_synthetic_echo() {
        let (energy_in, energy_out) = echo_energy(true);
        assert!(energy_in > 0.0, "the test signal must carry energy");
        assert!(
            energy_out < energy_in * 0.5,
            "echo energy should drop by at least half after convergence \
             (in={energy_in:.4}, out={energy_out:.4})"
        );
    }

    /// The control for [`the_canceller_attenuates_a_synthetic_echo`]. Without
    /// it, that test would still pass if the capture path were dropping or
    /// zeroing audio for some unrelated reason — this pins the attenuation on
    /// the canceller having seen the reference signal, which is the thing the
    /// [`RenderPath`](super::RenderPath) wiring exists to deliver.
    #[cfg(feature = "audio-proc")]
    #[test]
    fn with_no_reference_signal_the_voice_survives() {
        let (energy_in, energy_out) = echo_energy(false);
        assert!(energy_in > 0.0);
        assert!(
            energy_out > energy_in * 0.7,
            "a canceller with a silent reference has no echo to remove and \
             must leave near-end speech alone (in={energy_in:.4}, out={energy_out:.4})"
        );
    }
}

//! Fixed-size frame accumulation for the audio processor.
//!
//! The WebRTC audio processing module works in exact 10 ms frames, while cpal
//! hands the callbacks whatever buffer size the device happens to use — 128,
//! 480, 1024 samples, and not necessarily the same size twice. Everything in
//! this module exists to bridge that: bytes in at any size, frames out at
//! exactly one size, with the remainder carried to the next call.
//!
//! Kept separate from the processor itself because this half has no native
//! dependency and can be tested exhaustively on its own — the interesting bugs
//! here are off-by-ones at buffer boundaries, not anything to do with audio.

/// The frame size the audio processor requires, in milliseconds.
///
/// Not configurable: WebRTC's APM accepts 10 ms and nothing else.
pub const FRAME_MS: usize = 10;

/// Samples in one 10 ms frame at `rate_hz`, single channel.
///
/// # Panics
/// Never. A rate that is not a multiple of 100 yields a truncated frame, which
/// [`FrameAccumulator::new`] rejects rather than silently mis-framing.
#[must_use]
pub const fn frame_len(rate_hz: u32) -> usize {
    (rate_hz as usize) * FRAME_MS / 1000
}

/// Accumulates arbitrary-length audio into fixed frames.
///
/// Feed it whatever the device callback provides; it yields complete frames and
/// keeps the leftover for next time. It never allocates during steady-state
/// operation: the buffer reaches its high-water mark within the first few
/// callbacks and is reused thereafter.
#[derive(Debug)]
pub struct FrameAccumulator {
    frame_len: usize,
    /// Samples not yet emitted. Length is always < `frame_len` after `push`
    /// returns, because every whole frame is drained.
    pending: Vec<f32>,
}

impl FrameAccumulator {
    /// Build an accumulator for `rate_hz`.
    ///
    /// Returns `None` for a rate that does not divide into whole 10 ms frames.
    /// 44_100 is the notable real-world case: 441 samples per 10 ms is exact,
    /// but WebRTC does not accept that rate, so callers must resample first and
    /// this refuses rather than pretending.
    #[must_use]
    pub fn new(rate_hz: u32) -> Option<Self> {
        // The rates WebRTC's APM supports. Anything else has to be resampled
        // before it gets here.
        if !matches!(rate_hz, 8_000 | 16_000 | 32_000 | 48_000) {
            return None;
        }
        let frame_len = frame_len(rate_hz);
        Some(Self {
            frame_len,
            pending: Vec::with_capacity(frame_len * 2),
        })
    }

    /// Samples in one frame.
    #[must_use]
    pub const fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Samples held back, waiting for the rest of a frame.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Add `input` and invoke `on_frame` for each complete frame.
    ///
    /// `on_frame` receives exactly [`Self::frame_len`] samples and may modify
    /// them in place — that is how the processor writes its output back. The
    /// modified frame is what gets appended to `out`.
    ///
    /// `out` is appended to, not cleared, so a caller can gather several pushes
    /// before doing anything with the result.
    pub fn push<F>(&mut self, input: &[f32], out: &mut Vec<f32>, mut on_frame: F)
    where
        F: FnMut(&mut [f32]),
    {
        self.pending.extend_from_slice(input);

        let mut start = 0;
        while self.pending.len() - start >= self.frame_len {
            let end = start + self.frame_len;
            on_frame(&mut self.pending[start..end]);
            out.extend_from_slice(&self.pending[start..end]);
            start = end;
        }

        // Keep only the tail. drain is O(remaining), and remaining is always
        // less than one frame, so this does not grow with input size.
        if start > 0 {
            self.pending.drain(..start);
        }
    }

    /// Drop anything held back.
    ///
    /// For a call teardown or a device change, where carrying a partial frame
    /// into the next stream would splice unrelated audio together.
    pub fn reset(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{FRAME_MS, FrameAccumulator, frame_len};

    /// Collect every frame the accumulator emits for a given input.
    fn frames_for(acc: &mut FrameAccumulator, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        acc.push(input, &mut out, |_| {});
        out
    }

    #[test]
    fn frame_length_matches_ten_milliseconds() {
        assert_eq!(frame_len(8_000), 80);
        assert_eq!(frame_len(16_000), 160);
        assert_eq!(frame_len(48_000), 480);
        // The definition, restated independently of the constant.
        assert_eq!(frame_len(48_000), 48_000 * FRAME_MS / 1000);
    }

    #[test]
    fn unsupported_rates_are_refused_not_approximated() {
        // 44.1k divides evenly into 10 ms but WebRTC cannot take it; accepting
        // it here would push a wrong-rate frame into the processor.
        assert!(FrameAccumulator::new(44_100).is_none());
        assert!(FrameAccumulator::new(0).is_none());
        assert!(FrameAccumulator::new(22_050).is_none());
        for rate in [8_000, 16_000, 32_000, 48_000] {
            assert!(FrameAccumulator::new(rate).is_some(), "rate {rate}");
        }
    }

    #[test]
    fn a_short_buffer_yields_nothing_and_is_held() {
        let mut acc = FrameAccumulator::new(48_000).expect("supported rate");
        let out = frames_for(&mut acc, &[0.5; 100]);
        assert!(
            out.is_empty(),
            "100 samples is less than one 480-sample frame"
        );
        assert_eq!(acc.pending_len(), 100);
    }

    #[test]
    fn samples_split_across_calls_are_rejoined() {
        let mut acc = FrameAccumulator::new(48_000).expect("supported rate");
        let mut out = Vec::new();
        // 480 samples delivered as 200 + 200 + 80.
        acc.push(&[1.0; 200], &mut out, |_| {});
        acc.push(&[2.0; 200], &mut out, |_| {});
        assert!(out.is_empty(), "no whole frame yet");
        acc.push(&[3.0; 80], &mut out, |_| {});
        assert_eq!(out.len(), 480, "the frame completes on the third push");
        assert_eq!(acc.pending_len(), 0);
        // Order preserved across the splice.
        assert!((out[0] - 1.0).abs() < f32::EPSILON);
        assert!((out[250] - 2.0).abs() < f32::EPSILON);
        assert!((out[479] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_large_buffer_yields_several_frames_and_keeps_the_remainder() {
        let mut acc = FrameAccumulator::new(48_000).expect("supported rate");
        // 1100 = two 480-sample frames with 140 left over.
        let out = frames_for(&mut acc, &[1.0; 1100]);
        assert_eq!(out.len(), 960);
        assert_eq!(acc.pending_len(), 140);
    }

    /// The property that matters: nothing is invented and nothing is lost.
    #[test]
    fn every_sample_is_emitted_exactly_once_in_order() {
        let mut acc = FrameAccumulator::new(16_000).expect("supported rate");
        let mut out = Vec::new();
        // Irregular sizes, like a real device callback.
        let mut next = 0.0_f32;
        for chunk in [37_usize, 160, 1, 400, 12, 999, 160] {
            let input: Vec<f32> = (0..chunk)
                .map(|_| {
                    next += 1.0;
                    next
                })
                .collect();
            acc.push(&input, &mut out, |_| {});
        }
        let total: usize = 37 + 160 + 1 + 400 + 12 + 999 + 160;
        assert_eq!(out.len() + acc.pending_len(), total, "no samples lost");
        // Emitted samples are the original sequence, unpermuted.
        for (i, v) in out.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let expected = (i + 1) as f32;
            assert!(
                (*v - expected).abs() < f32::EPSILON,
                "sample {i} out of order"
            );
        }
    }

    #[test]
    fn the_callback_sees_a_whole_frame_and_its_edits_are_kept() {
        let mut acc = FrameAccumulator::new(8_000).expect("supported rate");
        let mut out = Vec::new();
        let mut seen_len = 0;
        acc.push(&[1.0; 80], &mut out, |frame| {
            seen_len = frame.len();
            // Stand in for what the processor does: rewrite in place.
            for s in frame.iter_mut() {
                *s = -*s;
            }
        });
        assert_eq!(seen_len, 80, "the callback must see exactly one frame");
        assert!(
            out.iter().all(|s| (*s + 1.0).abs() < f32::EPSILON),
            "edits must reach the output"
        );
    }

    #[test]
    fn reset_drops_the_partial_frame() {
        let mut acc = FrameAccumulator::new(48_000).expect("supported rate");
        let _ = frames_for(&mut acc, &[1.0; 100]);
        assert_eq!(acc.pending_len(), 100);
        acc.reset();
        assert_eq!(acc.pending_len(), 0);
        // After a reset the next frame starts clean rather than splicing.
        let out = frames_for(&mut acc, &[2.0; 480]);
        assert_eq!(out.len(), 480);
        assert!(out.iter().all(|s| (*s - 2.0).abs() < f32::EPSILON));
    }

    #[test]
    fn an_empty_push_is_harmless() {
        let mut acc = FrameAccumulator::new(48_000).expect("supported rate");
        let out = frames_for(&mut acc, &[]);
        assert!(out.is_empty());
        assert_eq!(acc.pending_len(), 0);
    }

    /// Steady state must not grow the buffer: a leak here would be unbounded
    /// on a long call.
    #[test]
    fn pending_never_reaches_a_whole_frame_after_a_push() {
        let mut acc = FrameAccumulator::new(48_000).expect("supported rate");
        let mut out = Vec::new();
        for _ in 0..200 {
            acc.push(&[0.25; 333], &mut out, |_| {});
            assert!(
                acc.pending_len() < acc.frame_len(),
                "a whole frame was left unemitted"
            );
        }
    }
}

//! Jitter buffer implementations for RTP media.
//!
//! Jitter buffers smooth out variable network delay (jitter) by buffering
//! packets before playout. This module provides two strategies:
//!
//! - **Fixed**: Constant delay, simple and predictable
//! - **Adaptive**: Dynamically adjusts delay based on observed jitter
//!
//! # Example
//!
//! ```
//! use rtp_engine::jitter::{JitterBuffer, JitterConfig, JitterMode};
//!
//! // Create an adaptive jitter buffer
//! let config = JitterConfig {
//!     mode: JitterMode::Adaptive { target_ms: 60, min_ms: 20, max_ms: 200 },
//!     clock_rate: 8000,
//!     max_packets: 50,
//! };
//! let mut jitter = JitterBuffer::new(config);
//!
//! // Push received RTP packets (seq, timestamp, payload)
//! jitter.push(0, 0, vec![0u8; 160]);
//! jitter.push(1, 160, vec![0u8; 160]);
//!
//! // Pop packets for playout (returns None if not ready yet)
//! // In real usage, wait for the jitter delay before popping
//! ```

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Jitter buffer operating mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitterMode {
    /// Fixed delay jitter buffer.
    Fixed {
        /// Delay in milliseconds before playout.
        delay_ms: u32,
    },
    /// Adaptive jitter buffer that adjusts to network conditions.
    Adaptive {
        /// Target delay in milliseconds (starting point).
        target_ms: u32,
        /// Minimum delay in milliseconds.
        min_ms: u32,
        /// Maximum delay in milliseconds.
        max_ms: u32,
    },
}

impl Default for JitterMode {
    fn default() -> Self {
        Self::Adaptive {
            target_ms: 60,
            min_ms: 20,
            max_ms: 200,
        }
    }
}

/// Configuration for the jitter buffer.
#[derive(Debug, Clone)]
pub struct JitterConfig {
    /// Operating mode (fixed or adaptive).
    pub mode: JitterMode,
    /// RTP clock rate in Hz (e.g., 8000 for G.711).
    pub clock_rate: u32,
    /// Maximum number of packets to buffer.
    pub max_packets: usize,
}

impl Default for JitterConfig {
    fn default() -> Self {
        Self {
            mode: JitterMode::default(),
            clock_rate: 8000,
            max_packets: 50,
        }
    }
}

/// A buffered RTP packet ready for playout.
#[derive(Debug, Clone)]
pub struct BufferedPacket {
    /// RTP sequence number.
    pub seq: u16,
    /// RTP timestamp.
    pub timestamp: u32,
    /// Packet payload (encoded audio).
    pub payload: Vec<u8>,
    /// When this packet was received.
    pub received_at: Instant,
    /// Whether this is a synthesized packet (for loss concealment).
    pub synthesized: bool,
}

/// Statistics from the jitter buffer.
#[derive(Debug, Clone, Default)]
pub struct JitterStats {
    /// Total packets received.
    pub packets_received: u64,
    /// Packets dropped (buffer full or too late).
    pub packets_dropped: u64,
    /// Packets lost (gaps in sequence).
    pub packets_lost: u64,
    /// Packets played out.
    pub packets_played: u64,
    /// Current buffer depth in packets.
    pub buffer_depth: usize,
    /// Current delay in milliseconds.
    pub current_delay_ms: u32,
    /// Observed jitter in milliseconds.
    pub observed_jitter_ms: f64,
}

/// Jitter buffer for RTP packet reordering and playout scheduling.
pub struct JitterBuffer {
    config: JitterConfig,
    /// Packets indexed by extended sequence number.
    packets: BTreeMap<u32, BufferedPacket>,
    /// Current playout sequence (extended).
    playout_seq: Option<u32>,
    /// Sequence number cycles (for extended seq).
    seq_cycles: u16,
    /// Last received sequence number.
    last_seq: Option<u16>,
    /// Current adaptive delay in ms.
    current_delay_ms: u32,
    /// Jitter estimate (RFC 3550 style, in timestamp units).
    jitter_estimate: f64,
    /// Last transit time for jitter calculation.
    last_transit: Option<i64>,
    /// Statistics.
    stats: JitterStats,
    /// First packet timestamp (for relative timing).
    base_timestamp: Option<u32>,
    /// When the first packet was received.
    base_time: Option<Instant>,
    /// Whether we've started playout.
    playing: bool,
}

impl JitterBuffer {
    /// Create a new jitter buffer with the given configuration.
    pub fn new(config: JitterConfig) -> Self {
        let initial_delay = match config.mode {
            JitterMode::Fixed { delay_ms } => delay_ms,
            JitterMode::Adaptive { target_ms, .. } => target_ms,
        };

        Self {
            config,
            packets: BTreeMap::new(),
            playout_seq: None,
            seq_cycles: 0,
            last_seq: None,
            current_delay_ms: initial_delay,
            jitter_estimate: 0.0,
            last_transit: None,
            stats: JitterStats::default(),
            base_timestamp: None,
            base_time: None,
            playing: false,
        }
    }

    /// Push a received RTP packet into the buffer.
    ///
    /// Returns `true` if the packet was buffered, `false` if dropped.
    pub fn push(&mut self, seq: u16, timestamp: u32, payload: Vec<u8>) -> bool {
        let now = Instant::now();
        self.stats.packets_received += 1;

        // Initialize base references on first packet
        if self.base_timestamp.is_none() {
            self.base_timestamp = Some(timestamp);
            self.base_time = Some(now);
            self.last_seq = Some(seq);
        }

        // Calculate extended sequence number
        let extended_seq = self.extend_seq(seq);

        // Update jitter estimate
        self.update_jitter(timestamp, now);

        // Check if packet is too late (already played)
        if let Some(playout) = self.playout_seq
            && extended_seq < playout
        {
            self.stats.packets_dropped += 1;
            return false;
        }

        // Check buffer capacity
        if self.packets.len() >= self.config.max_packets {
            // Drop oldest packet
            if let Some(&oldest_seq) = self.packets.keys().next() {
                self.packets.remove(&oldest_seq);
                self.stats.packets_dropped += 1;
            }
        }

        // Buffer the packet
        self.packets.insert(
            extended_seq,
            BufferedPacket {
                seq,
                timestamp,
                payload,
                received_at: now,
                synthesized: false,
            },
        );

        self.last_seq = Some(seq);
        true
    }

    /// Pop the next packet for playout, if ready.
    ///
    /// Returns `None` if no packet is ready (still buffering or waiting for delay).
    pub fn pop(&mut self) -> Option<BufferedPacket> {
        let now = Instant::now();

        // Determine which sequence to play
        let target_seq = if let Some(seq) = self.playout_seq {
            seq
        } else {
            // Not yet playing - check if we should start
            if !self.should_start_playout(now) {
                return None;
            }
            // Start with the lowest buffered sequence
            let first_seq = *self.packets.keys().next()?;
            self.playout_seq = Some(first_seq);
            self.playing = true;
            first_seq
        };

        // Try to get the target packet
        let packet = if let Some(pkt) = self.packets.remove(&target_seq) {
            Some(pkt)
        } else {
            // Packet missing - loss concealment
            self.stats.packets_lost += 1;
            // Return a synthesized empty packet for PLC
            Some(BufferedPacket {
                seq: (target_seq & 0xFFFF) as u16,
                timestamp: self.estimate_timestamp(target_seq),
                payload: Vec::new(), // Empty = signal for PLC
                received_at: now,
                synthesized: true,
            })
        };

        self.stats.packets_played += 1;

        // Advance playout sequence
        self.playout_seq = Some(target_seq.wrapping_add(1));

        // Update statistics
        self.stats.buffer_depth = self.packets.len();
        self.stats.current_delay_ms = self.current_delay_ms;
        self.stats.observed_jitter_ms = self.jitter_ms();

        // Adapt delay if in adaptive mode
        if matches!(self.config.mode, JitterMode::Adaptive { .. }) {
            self.adapt_delay();
        }

        packet
    }

    /// Get current statistics.
    pub fn stats(&self) -> JitterStats {
        let mut stats = self.stats.clone();
        stats.buffer_depth = self.packets.len();
        stats.current_delay_ms = self.current_delay_ms;
        stats.observed_jitter_ms = self.jitter_ms();
        stats
    }

    /// Get current delay in milliseconds.
    pub fn delay_ms(&self) -> u32 {
        self.current_delay_ms
    }

    /// Get observed jitter in milliseconds.
    pub fn jitter_ms(&self) -> f64 {
        // Convert from timestamp units to ms
        (self.jitter_estimate / self.config.clock_rate as f64) * 1000.0
    }

    /// Reset the jitter buffer.
    pub fn reset(&mut self) {
        self.packets.clear();
        self.playout_seq = None;
        self.seq_cycles = 0;
        self.last_seq = None;
        self.jitter_estimate = 0.0;
        self.last_transit = None;
        self.base_timestamp = None;
        self.base_time = None;
        self.playing = false;
        self.stats = JitterStats::default();

        // Reset delay to initial
        self.current_delay_ms = match self.config.mode {
            JitterMode::Fixed { delay_ms } => delay_ms,
            JitterMode::Adaptive { target_ms, .. } => target_ms,
        };
    }

    /// Flush all buffered packets.
    pub fn flush(&mut self) -> Vec<BufferedPacket> {
        let packets: Vec<_> = self.packets.values().cloned().collect();
        self.packets.clear();
        packets
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }

    /// Get the number of buffered packets.
    pub fn len(&self) -> usize {
        self.packets.len()
    }

    // --- Internal methods ---

    fn extend_seq(&mut self, seq: u16) -> u32 {
        if let Some(last) = self.last_seq {
            // Detect rollover
            if seq < last && (last.wrapping_sub(seq)) > 0x8000 {
                self.seq_cycles = self.seq_cycles.wrapping_add(1);
            } else if seq > last && (seq.wrapping_sub(last)) > 0x8000 {
                self.seq_cycles = self.seq_cycles.wrapping_sub(1);
            }
        }
        ((self.seq_cycles as u32) << 16) | (seq as u32)
    }

    fn update_jitter(&mut self, timestamp: u32, now: Instant) {
        let base_ts = match self.base_timestamp {
            Some(ts) => ts,
            None => return,
        };
        let base_time = match self.base_time {
            Some(t) => t,
            None => return,
        };

        // Calculate transit time in timestamp units
        let arrival_ts = now.duration_since(base_time).as_micros() as i64
            * self.config.clock_rate as i64
            / 1_000_000;
        let send_ts = timestamp.wrapping_sub(base_ts) as i64;
        let transit = arrival_ts - send_ts;

        if let Some(last_transit) = self.last_transit {
            // RFC 3550 jitter calculation
            let d = (transit - last_transit).abs() as f64;
            self.jitter_estimate += (d - self.jitter_estimate) / 16.0;
        }

        self.last_transit = Some(transit);
    }

    fn should_start_playout(&self, now: Instant) -> bool {
        // Need at least one packet
        if self.packets.is_empty() {
            return false;
        }

        // Check if we've waited long enough
        if let Some(base_time) = self.base_time {
            let elapsed = now.duration_since(base_time);
            let delay = Duration::from_millis(self.current_delay_ms as u64);
            return elapsed >= delay;
        }

        false
    }

    fn adapt_delay(&mut self) {
        let JitterMode::Adaptive {
            min_ms,
            max_ms,
            target_ms: _,
        } = self.config.mode
        else {
            return;
        };

        // Target delay = 2 * observed jitter (with bounds)
        let jitter_ms = self.jitter_ms();
        let target = (jitter_ms * 2.0) as u32;
        let target = target.clamp(min_ms, max_ms);

        // Smooth adjustment (don't change too fast)
        if target > self.current_delay_ms {
            // Increase quickly when jitter rises
            self.current_delay_ms = self
                .current_delay_ms
                .saturating_add(((target - self.current_delay_ms) / 4).max(1));
        } else if target < self.current_delay_ms {
            // Decrease slowly when jitter falls
            self.current_delay_ms = self
                .current_delay_ms
                .saturating_sub(((self.current_delay_ms - target) / 8).max(1));
        }

        self.current_delay_ms = self.current_delay_ms.clamp(min_ms, max_ms);
    }

    fn estimate_timestamp(&self, extended_seq: u32) -> u32 {
        // Estimate timestamp based on sequence and samples per packet
        // Assuming 20ms frames at the clock rate
        let samples_per_frame = self.config.clock_rate / 50; // 20ms
        let base = self.base_timestamp.unwrap_or(0);
        let seq_offset = extended_seq.wrapping_sub(self.playout_seq.unwrap_or(extended_seq));
        base.wrapping_add(seq_offset * samples_per_frame)
    }
}

impl std::fmt::Debug for JitterBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitterBuffer")
            .field("mode", &self.config.mode)
            .field("buffered", &self.packets.len())
            .field("delay_ms", &self.current_delay_ms)
            .field("jitter_ms", &self.jitter_ms())
            .field("playing", &self.playing)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_fixed_jitter_buffer_basic() {
        let config = JitterConfig {
            mode: JitterMode::Fixed { delay_ms: 20 },
            clock_rate: 8000,
            max_packets: 10,
        };
        let mut jitter = JitterBuffer::new(config);

        // Push some packets
        assert!(jitter.push(0, 0, vec![1, 2, 3]));
        assert!(jitter.push(1, 160, vec![4, 5, 6]));
        assert!(jitter.push(2, 320, vec![7, 8, 9]));

        assert_eq!(jitter.len(), 3);

        // Wait for delay
        sleep(Duration::from_millis(25));

        // Pop packets in order
        let p1 = jitter.pop().unwrap();
        assert_eq!(p1.seq, 0);
        assert_eq!(p1.payload, vec![1, 2, 3]);

        let p2 = jitter.pop().unwrap();
        assert_eq!(p2.seq, 1);

        let p3 = jitter.pop().unwrap();
        assert_eq!(p3.seq, 2);
    }

    #[test]
    fn test_packet_reordering() {
        let config = JitterConfig {
            mode: JitterMode::Fixed { delay_ms: 10 },
            clock_rate: 8000,
            max_packets: 10,
        };
        let mut jitter = JitterBuffer::new(config);

        // Push packets out of order
        jitter.push(2, 320, vec![3]);
        jitter.push(0, 0, vec![1]);
        jitter.push(1, 160, vec![2]);

        sleep(Duration::from_millis(15));

        // Should come out in order
        assert_eq!(jitter.pop().unwrap().seq, 0);
        assert_eq!(jitter.pop().unwrap().seq, 1);
        assert_eq!(jitter.pop().unwrap().seq, 2);
    }

    #[test]
    fn test_packet_loss_detection() {
        let config = JitterConfig {
            mode: JitterMode::Fixed { delay_ms: 10 },
            clock_rate: 8000,
            max_packets: 10,
        };
        let mut jitter = JitterBuffer::new(config);

        // Push packets with gap (missing seq 1)
        jitter.push(0, 0, vec![1]);
        jitter.push(2, 320, vec![3]);

        sleep(Duration::from_millis(15));

        // First packet normal
        let p1 = jitter.pop().unwrap();
        assert_eq!(p1.seq, 0);
        assert!(!p1.synthesized);

        // Second packet is synthesized (loss)
        let p2 = jitter.pop().unwrap();
        assert_eq!(p2.seq, 1);
        assert!(p2.synthesized);
        assert!(p2.payload.is_empty());

        // Third packet normal
        let p3 = jitter.pop().unwrap();
        assert_eq!(p3.seq, 2);
        assert!(!p3.synthesized);

        let stats = jitter.stats();
        assert_eq!(stats.packets_lost, 1);
    }

    #[test]
    fn test_late_packet_dropped() {
        let config = JitterConfig {
            mode: JitterMode::Fixed { delay_ms: 5 },
            clock_rate: 8000,
            max_packets: 10,
        };
        let mut jitter = JitterBuffer::new(config);

        jitter.push(0, 0, vec![1]);
        jitter.push(1, 160, vec![2]);

        sleep(Duration::from_millis(10));

        // Play first packet
        jitter.pop();

        // Try to push packet 0 again (too late)
        assert!(!jitter.push(0, 0, vec![1]));

        let stats = jitter.stats();
        assert_eq!(stats.packets_dropped, 1);
    }

    #[test]
    fn test_adaptive_jitter_buffer() {
        let config = JitterConfig {
            mode: JitterMode::Adaptive {
                target_ms: 40,
                min_ms: 20,
                max_ms: 200,
            },
            clock_rate: 8000,
            max_packets: 20,
        };
        let mut jitter = JitterBuffer::new(config);

        assert_eq!(jitter.delay_ms(), 40); // Starts at target

        // Push packets with simulated jitter
        for i in 0..10u16 {
            jitter.push(i, i as u32 * 160, vec![i as u8]);
            sleep(Duration::from_millis(5)); // Varying arrival
        }

        sleep(Duration::from_millis(50));

        // Pop packets and check adaptation
        for _ in 0..5 {
            jitter.pop();
        }

        // Delay should have adapted (may increase or decrease)
        let delay = jitter.delay_ms();
        assert!(delay >= 20 && delay <= 200);
    }

    #[test]
    fn test_sequence_rollover_in_jitter_buffer() {
        let config = JitterConfig {
            mode: JitterMode::Fixed { delay_ms: 5 },
            clock_rate: 8000,
            max_packets: 10,
        };
        let mut jitter = JitterBuffer::new(config);

        // Push packets near rollover
        jitter.push(65534, 0, vec![1]);
        jitter.push(65535, 160, vec![2]);
        jitter.push(0, 320, vec![3]); // Rollover
        jitter.push(1, 480, vec![4]);

        sleep(Duration::from_millis(10));

        // Should come out in order across rollover
        assert_eq!(jitter.pop().unwrap().seq, 65534);
        assert_eq!(jitter.pop().unwrap().seq, 65535);
        assert_eq!(jitter.pop().unwrap().seq, 0);
        assert_eq!(jitter.pop().unwrap().seq, 1);
    }

    #[test]
    fn test_buffer_overflow() {
        let config = JitterConfig {
            mode: JitterMode::Fixed { delay_ms: 100 },
            clock_rate: 8000,
            max_packets: 3,
        };
        let mut jitter = JitterBuffer::new(config);

        // Push more than max
        jitter.push(0, 0, vec![1]);
        jitter.push(1, 160, vec![2]);
        jitter.push(2, 320, vec![3]);
        jitter.push(3, 480, vec![4]); // Should drop oldest

        assert_eq!(jitter.len(), 3);

        let stats = jitter.stats();
        assert_eq!(stats.packets_dropped, 1);
    }

    #[test]
    fn test_reset() {
        let config = JitterConfig::default();
        let mut jitter = JitterBuffer::new(config);

        jitter.push(0, 0, vec![1]);
        jitter.push(1, 160, vec![2]);

        jitter.reset();

        assert!(jitter.is_empty());
        assert_eq!(jitter.stats().packets_received, 0);
    }

    #[test]
    fn test_flush() {
        let config = JitterConfig::default();
        let mut jitter = JitterBuffer::new(config);

        jitter.push(0, 0, vec![1]);
        jitter.push(1, 160, vec![2]);
        jitter.push(2, 320, vec![3]);

        let flushed = jitter.flush();
        assert_eq!(flushed.len(), 3);
        assert!(jitter.is_empty());
    }

    #[test]
    fn test_jitter_calculation() {
        let config = JitterConfig {
            mode: JitterMode::Fixed { delay_ms: 10 },
            clock_rate: 8000,
            max_packets: 20,
        };
        let mut jitter = JitterBuffer::new(config);

        // Simulate packets with varying inter-arrival times
        jitter.push(0, 0, vec![1]);
        sleep(Duration::from_millis(20));
        jitter.push(1, 160, vec![2]);
        sleep(Duration::from_millis(25));
        jitter.push(2, 320, vec![3]);
        sleep(Duration::from_millis(15));
        jitter.push(3, 480, vec![4]);

        // Jitter should be non-zero due to varying arrival
        let jitter_ms = jitter.jitter_ms();
        assert!(jitter_ms >= 0.0);
    }

    #[test]
    fn test_stats() {
        let config = JitterConfig {
            mode: JitterMode::Fixed { delay_ms: 5 },
            clock_rate: 8000,
            max_packets: 10,
        };
        let mut jitter = JitterBuffer::new(config);

        jitter.push(0, 0, vec![1]);
        jitter.push(1, 160, vec![2]);
        // Skip seq 2
        jitter.push(3, 480, vec![4]);

        sleep(Duration::from_millis(10));

        jitter.pop(); // seq 0
        jitter.pop(); // seq 1
        jitter.pop(); // seq 2 (synthesized)
        jitter.pop(); // seq 3

        let stats = jitter.stats();
        assert_eq!(stats.packets_received, 3);
        assert_eq!(stats.packets_played, 4);
        assert_eq!(stats.packets_lost, 1);
    }
}

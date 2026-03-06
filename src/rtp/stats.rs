//! RTP/RTCP statistics tracking.
//!
//! Implements RFC 3550 extended sequence number tracking with proper
//! rollover handling for 16-bit sequence numbers.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// RTP/RTCP statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct RtpStats {
    /// Total RTP packets sent.
    pub packets_sent: u64,
    /// Total RTP packets received.
    pub packets_received: u64,
    /// Total payload bytes sent.
    pub bytes_sent: u64,
    /// Total payload bytes received.
    pub bytes_received: u64,
    /// Estimated packets lost.
    pub packets_lost: u64,
    /// Interarrival jitter in milliseconds.
    pub jitter_ms: f64,
    /// Codec name.
    pub codec_name: String,
    /// Extended highest sequence number (includes rollover cycles).
    pub extended_highest_seq: u32,
    /// Number of sequence number rollovers.
    pub seq_cycles: u16,
}

/// Thread-safe counters for tracking RTP statistics.
#[derive(Clone)]
pub struct RtpCounters {
    /// Packets sent.
    pub packets_sent: Arc<AtomicU64>,
    /// Packets received.
    pub packets_received: Arc<AtomicU64>,
    /// Bytes sent.
    pub bytes_sent: Arc<AtomicU64>,
    /// Bytes received.
    pub bytes_received: Arc<AtomicU64>,
    /// Packets lost.
    pub packets_lost: Arc<AtomicU64>,
    /// Jitter in microseconds.
    pub jitter_us: Arc<AtomicU64>,
    /// Codec name.
    pub codec_name: String,
    /// Extended highest sequence number (upper 16 bits = cycles, lower 16 = seq).
    pub highest_seq: Arc<AtomicU32>,
    /// Expected packets based on sequence numbers.
    pub expected_packets: Arc<AtomicU64>,
    /// Whether we've received the first packet (to initialize tracking).
    initialized: Arc<AtomicBool>,
    /// Base (first) sequence number received.
    base_seq: Arc<AtomicU32>,
}

impl RtpCounters {
    /// Create a new set of counters.
    pub fn new(codec_name: &str) -> Self {
        Self {
            packets_sent: Arc::new(AtomicU64::new(0)),
            packets_received: Arc::new(AtomicU64::new(0)),
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            packets_lost: Arc::new(AtomicU64::new(0)),
            jitter_us: Arc::new(AtomicU64::new(0)),
            codec_name: codec_name.to_string(),
            highest_seq: Arc::new(AtomicU32::new(0)),
            expected_packets: Arc::new(AtomicU64::new(0)),
            initialized: Arc::new(AtomicBool::new(false)),
            base_seq: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Take a snapshot of the current statistics.
    pub fn snapshot(&self) -> RtpStats {
        let received = self.packets_received.load(Ordering::Relaxed);
        let expected = self.expected_packets.load(Ordering::Relaxed);
        let lost = expected.saturating_sub(received);
        self.packets_lost.store(lost, Ordering::Relaxed);
        let highest = self.highest_seq.load(Ordering::Relaxed);

        RtpStats {
            packets_sent: self.packets_sent.load(Ordering::Relaxed),
            packets_received: received,
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            packets_lost: lost,
            jitter_ms: self.jitter_us.load(Ordering::Relaxed) as f64 / 1000.0,
            codec_name: self.codec_name.clone(),
            extended_highest_seq: highest,
            seq_cycles: (highest >> 16) as u16,
        }
    }

    /// Record a sent packet.
    pub fn record_sent(&self, bytes: u64) {
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a received packet with proper sequence rollover handling.
    ///
    /// This implements RFC 3550 Appendix A.1 extended sequence number algorithm.
    pub fn record_received(&self, bytes: u64, seq: u16) {
        self.packets_received.fetch_add(1, Ordering::Relaxed);
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);

        if !self.initialized.swap(true, Ordering::Relaxed) {
            // First packet - initialize tracking
            self.base_seq.store(seq as u32, Ordering::Relaxed);
            self.highest_seq.store(seq as u32, Ordering::Relaxed);
            self.expected_packets.store(1, Ordering::Relaxed);
            return;
        }

        let prev_extended = self.highest_seq.load(Ordering::Relaxed);
        let prev_seq = (prev_extended & 0xFFFF) as u16;
        let cycles = prev_extended >> 16;

        // RFC 3550: detect rollover by checking if sequence wrapped
        let new_cycles = if seq < prev_seq && (prev_seq.wrapping_sub(seq)) > 0x8000 {
            // Sequence wrapped forward (65535 -> 0)
            cycles.wrapping_add(1)
        } else if seq > prev_seq && (seq.wrapping_sub(prev_seq)) > 0x8000 {
            // Late/reordered packet from before rollover
            cycles.wrapping_sub(1)
        } else {
            cycles
        };

        let new_extended = (new_cycles << 16) | (seq as u32);

        // Update if this is a higher extended sequence number
        if new_extended > prev_extended || (new_cycles > cycles) {
            self.highest_seq.store(new_extended, Ordering::Relaxed);

            // Update expected packets count
            let base = self.base_seq.load(Ordering::Relaxed);
            let expected = new_extended.wrapping_sub(base).wrapping_add(1) as u64;
            self.expected_packets.store(expected, Ordering::Relaxed);
        }
    }

    /// Update jitter calculation (RFC 3550 algorithm).
    pub fn update_jitter(&self, transit_diff_us: u64) {
        let prev_jitter = self.jitter_us.load(Ordering::Relaxed) as f64;
        let d = transit_diff_us as f64;
        let new_jitter = prev_jitter + (d - prev_jitter) / 16.0;
        self.jitter_us.store(new_jitter as u64, Ordering::Relaxed);
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.packets_sent.store(0, Ordering::Relaxed);
        self.packets_received.store(0, Ordering::Relaxed);
        self.bytes_sent.store(0, Ordering::Relaxed);
        self.bytes_received.store(0, Ordering::Relaxed);
        self.packets_lost.store(0, Ordering::Relaxed);
        self.jitter_us.store(0, Ordering::Relaxed);
        self.highest_seq.store(0, Ordering::Relaxed);
        self.expected_packets.store(0, Ordering::Relaxed);
        self.initialized.store(false, Ordering::Relaxed);
        self.base_seq.store(0, Ordering::Relaxed);
    }

    /// Get the extended highest sequence number (cycles << 16 | seq).
    pub fn extended_highest_seq(&self) -> u32 {
        self.highest_seq.load(Ordering::Relaxed)
    }

    /// Get the number of sequence cycles (rollovers).
    pub fn seq_cycles(&self) -> u16 {
        (self.highest_seq.load(Ordering::Relaxed) >> 16) as u16
    }
}

impl Default for RtpCounters {
    fn default() -> Self {
        Self::new("unknown")
    }
}

impl std::fmt::Debug for RtpCounters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtpCounters")
            .field("codec", &self.codec_name)
            .field("sent", &self.packets_sent.load(Ordering::Relaxed))
            .field("received", &self.packets_received.load(Ordering::Relaxed))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counters_basic() {
        let counters = RtpCounters::new("PCMU");

        counters.record_sent(172);
        counters.record_sent(172);
        counters.record_received(172, 1);
        counters.record_received(172, 2);

        let stats = counters.snapshot();
        assert_eq!(stats.packets_sent, 2);
        assert_eq!(stats.packets_received, 2);
        assert_eq!(stats.bytes_sent, 344);
        assert_eq!(stats.bytes_received, 344);
        assert_eq!(stats.codec_name, "PCMU");
    }

    #[test]
    fn test_jitter_calculation() {
        let counters = RtpCounters::new("PCMU");

        // Simulate varying transit times
        counters.update_jitter(1000);
        counters.update_jitter(2000);
        counters.update_jitter(500);

        let stats = counters.snapshot();
        assert!(stats.jitter_ms > 0.0);
    }

    #[test]
    fn test_reset() {
        let counters = RtpCounters::new("PCMU");

        counters.record_sent(100);
        counters.record_received(100, 1);
        counters.reset();

        let stats = counters.snapshot();
        assert_eq!(stats.packets_sent, 0);
        assert_eq!(stats.packets_received, 0);
    }

    #[test]
    fn test_sequence_rollover_forward() {
        let counters = RtpCounters::new("PCMU");

        // Start near the rollover point
        counters.record_received(100, 65534);
        assert_eq!(counters.seq_cycles(), 0);
        assert_eq!(counters.extended_highest_seq(), 65534);

        counters.record_received(100, 65535);
        assert_eq!(counters.seq_cycles(), 0);
        assert_eq!(counters.extended_highest_seq(), 65535);

        // Rollover: 65535 -> 0
        counters.record_received(100, 0);
        assert_eq!(counters.seq_cycles(), 1);
        assert_eq!(counters.extended_highest_seq(), 1 << 16); // cycle 1, seq 0

        counters.record_received(100, 1);
        assert_eq!(counters.seq_cycles(), 1);
        assert_eq!(counters.extended_highest_seq(), (1 << 16) | 1);

        // Continue normally in cycle 1
        counters.record_received(100, 2);
        counters.record_received(100, 3);
        assert_eq!(counters.seq_cycles(), 1);
        assert_eq!(counters.extended_highest_seq(), (1 << 16) | 3);
    }

    #[test]
    fn test_second_rollover_sequential() {
        let counters = RtpCounters::new("PCMU");

        // Start in cycle 0 near rollover
        counters.record_received(100, 65534);
        counters.record_received(100, 65535);
        counters.record_received(100, 0); // -> cycle 1
        assert_eq!(counters.seq_cycles(), 1);

        // Progress sequentially through cycle 1
        // (In real RTP, packets arrive sequentially)
        for seq in 1u16..=65535 {
            counters.record_received(100, seq);
        }
        // Now rollover again
        counters.record_received(100, 0); // -> cycle 2
        assert_eq!(counters.seq_cycles(), 2);
        assert_eq!(counters.extended_highest_seq(), 2 << 16); // cycle 2, seq 0
    }

    #[test]
    fn test_small_gap_near_rollover() {
        let counters = RtpCounters::new("PCMU");

        // Test small gaps (realistic packet loss) near rollover
        counters.record_received(100, 65530);
        counters.record_received(100, 65531);
        // Skip 65532 (lost)
        counters.record_received(100, 65533);
        counters.record_received(100, 65534);
        counters.record_received(100, 65535);
        // Rollover
        counters.record_received(100, 0);
        assert_eq!(counters.seq_cycles(), 1);
        // Skip 1 (lost)
        counters.record_received(100, 2);
        counters.record_received(100, 3);

        assert_eq!(counters.seq_cycles(), 1);
        assert_eq!(counters.extended_highest_seq(), (1 << 16) | 3);
    }

    #[test]
    fn test_sequence_reorder_near_rollover() {
        let counters = RtpCounters::new("PCMU");

        // Receive packet 65534
        counters.record_received(100, 65534);
        assert_eq!(counters.seq_cycles(), 0);

        // Receive packet 0 (rollover)
        counters.record_received(100, 0);
        assert_eq!(counters.seq_cycles(), 1);

        // Late arrival of 65535 from before rollover
        // Should not increment cycles further
        counters.record_received(100, 65535);
        assert_eq!(counters.seq_cycles(), 1);

        // Continue normally
        counters.record_received(100, 1);
        counters.record_received(100, 2);
        assert_eq!(counters.extended_highest_seq(), (1 << 16) | 2);
    }

    #[test]
    fn test_expected_packets_with_rollover() {
        let counters = RtpCounters::new("PCMU");

        // Start at 65530
        counters.record_received(100, 65530);

        // Go to 5 (across rollover)
        for seq in 65531..=65535 {
            counters.record_received(100, seq);
        }
        for seq in 0..=5 {
            counters.record_received(100, seq);
        }

        let stats = counters.snapshot();
        // Expected: 65530 to 65535 (6) + 0 to 5 (6) = 12 packets
        // But we also count from base_seq to highest, so:
        // base = 65530, highest = (1 << 16) | 5 = 65541
        // expected = 65541 - 65530 + 1 = 12
        assert_eq!(stats.packets_received, 12);
        assert_eq!(stats.extended_highest_seq, (1 << 16) | 5);
    }

    #[test]
    fn test_multiple_rollovers() {
        let counters = RtpCounters::new("PCMU");

        counters.record_received(100, 0);

        // Simulate 3 full cycles
        for cycle in 0..3 {
            for seq in 1..=65535u16 {
                counters.record_received(100, seq);
            }
            counters.record_received(100, 0);
            assert_eq!(
                counters.seq_cycles(),
                cycle + 1,
                "After cycle {}, expected {} cycles",
                cycle,
                cycle + 1
            );
        }

        assert_eq!(counters.seq_cycles(), 3);
    }

    #[test]
    fn test_packet_loss_calculation() {
        let counters = RtpCounters::new("PCMU");

        // Receive packets 0, 1, 2, 5, 6 (missing 3, 4)
        counters.record_received(100, 0);
        counters.record_received(100, 1);
        counters.record_received(100, 2);
        counters.record_received(100, 5);
        counters.record_received(100, 6);

        let stats = counters.snapshot();
        assert_eq!(stats.packets_received, 5);
        // Expected: 0 to 6 = 7 packets
        // Lost: 7 - 5 = 2
        assert_eq!(stats.packets_lost, 2);
    }
}

//! ITU-T G.729 Annex A (CS-ACELP, 8 kbit/s) with Annex B (VAD, DTX and
//! comfort noise), in pure Rust.
//!
//! A port of the ITU-T G.729A + Annex B fixed-point reference
//! implementation (ITU-T G.729 software package release 3), bit-exact with
//! it: the tests in `tests.rs` run the ITU-T Annex A and Annex B test
//! vectors through both the encoder and the decoder. G.729A's bitstream is
//! the same as full G.729's, so it interoperates with any G.729 or G.729A
//! peer.
//!
//! # Framing
//!
//! - **Sample rate**: 8000 Hz
//! - **Frame size**: 10 ms (80 samples), 10 bytes
//! - **Bitrate**: 8 kbit/s
//! - **RTP payload type**: 18, with the 80 bits of each frame packed MSB
//!   first (RFC 3551 §4.5.6). A packet may carry several frames.
//!
//! # Annex B
//!
//! The decoder always understands Annex B: a 2-byte SID frame (15 bits and
//! a zero pad, at the end of a packet) updates the comfort noise, and
//! [`AudioDecoder::conceal_loss`] continues that noise through the frames a
//! DTX sender leaves out. After speech, `conceal_loss` is G.729's ordinary
//! frame-erasure concealment.
//!
//! The encoder sends speech frames only unless built with
//! [`G729Encoder::with_dtx`]. With DTX it sends a SID when the background
//! noise changes and nothing otherwise; only offer that to a peer that
//! negotiated `annexb=yes` (the SDP default for G.729).
//!
//! # Provenance
//!
//! The algorithm, constants and tables come from the reference source,
//! Copyright (c) 1996 AT&T, France Telecom, NTT, Universite de Sherbrooke,
//! distributed by the ITU-T with Recommendation G.729. The G.729 patents
//! have expired.

mod acelp;
mod basic_op;
mod decoder;
mod dsp;
mod dtx;
mod encoder;
mod gain;
mod lpc;
mod lsp;
mod pitch;
mod postfilter;
mod tables;
#[cfg(test)]
mod tests;
mod vad;

use super::{AudioDecoder, AudioEncoder, CodecType, EncodedPacket};
use decoder::Frame;
use encoder::Encoded;
use tables::{BITSNO, BITSNO2};

/// Samples in one 10 ms frame.
pub(crate) const FRAME_SAMPLES: usize = L_FRAME;
/// Bytes in one encoded frame.
pub(crate) const FRAME_BYTES: usize = 10;
/// An Annex B comfort-noise (SID) frame.
const SID_BYTES: usize = 2;

const L_TOTAL: usize = 240;
const L_WINDOW: usize = 240;
const L_NEXT: usize = 40;
const L_FRAME: usize = 80;
const L_SUBFR: usize = 40;
const M: usize = 10;
const MP1: usize = M + 1;
const NC: usize = M / 2;
const PIT_MIN: usize = 20;
const PIT_MAX: usize = 143;
const L_INTERPOL: usize = 10 + 1;
const GAMMA1: i16 = 24576;
const PRM_SIZE: usize = 11;
const GRID_POINTS: usize = 50;
const SHARPMAX: i16 = 13017;
const SHARPMIN: i16 = 3277;

/// Initial LSPs of both encoder and decoder.
const LSP_INIT: [i16; M] = [
    30000, 26000, 21000, 15000, 8000, 0, -8000, -15000, -21000, -26000,
];

/// Pack the 11 parameters into a 10-byte RTP payload frame.
fn pack(prm: &[i16; PRM_SIZE], out: &mut Vec<u8>) {
    let mut acc: u128 = 0;
    for (&p, &bits) in prm.iter().zip(&BITSNO) {
        acc = (acc << bits) | u128::from(p as u16 & ((1u16 << bits) - 1));
    }
    out.extend_from_slice(&acc.to_be_bytes()[16 - FRAME_BYTES..]);
}

/// Pack the 4 SID parameters into 2 bytes: 15 bits, then a zero pad bit.
fn pack_sid(prm: [i16; 4], out: &mut Vec<u8>) {
    let mut acc: u16 = 0;
    for (&p, &bits) in prm.iter().zip(&BITSNO2) {
        acc = (acc << bits) | (p as u16 & ((1u16 << bits) - 1));
    }
    out.extend_from_slice(&(acc << 1).to_be_bytes());
}

fn unpack_sid(frame: &[u8]) -> [i16; 4] {
    let acc = u16::from_be_bytes([frame[0], frame[1]]) >> 1;
    let mut prm = [0i16; 4];
    let mut shift = 15u32;
    for (p, &bits) in prm.iter_mut().zip(&BITSNO2) {
        shift -= bits as u32;
        *p = ((acc >> shift) & ((1u16 << bits) - 1)) as i16;
    }
    prm
}

/// Unpack a 10-byte RTP payload frame into the 11 parameters.
fn unpack(frame: &[u8]) -> [i16; PRM_SIZE] {
    let mut bytes = [0u8; 16];
    bytes[16 - FRAME_BYTES..].copy_from_slice(&frame[..FRAME_BYTES]);
    let acc = u128::from_be_bytes(bytes);
    let mut prm = [0i16; PRM_SIZE];
    let mut shift = FRAME_BYTES as u32 * 8;
    for (p, &bits) in prm.iter_mut().zip(&BITSNO) {
        shift -= bits as u32;
        *p = ((acc >> shift) & ((1u128 << bits) - 1)) as i16;
    }
    prm
}

/// G.729 Annex A encoder, optionally with Annex B DTX.
pub struct G729Encoder {
    inner: Box<encoder::Encoder>,
}

impl G729Encoder {
    /// Create an encoder that sends a speech frame for every 10 ms. Any
    /// G.729 peer accepts its output.
    pub fn new() -> Self {
        Self {
            inner: Box::new(encoder::Encoder::new(false)),
        }
    }

    /// Create an encoder with Annex B discontinuous transmission: during
    /// silence it sends a 2-byte SID frame when the background noise
    /// changes, and nothing otherwise. Only for peers that accept Annex B
    /// (`annexb=yes`, the SDP default).
    pub fn with_dtx() -> Self {
        Self {
            inner: Box::new(encoder::Encoder::new(true)),
        }
    }
}

impl Default for G729Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEncoder for G729Encoder {
    /// Encode whole 10 ms frames of 8 kHz PCM; a trailing partial frame is
    /// left unconsumed. With DTX, a SID frame adds 2 bytes and an
    /// untransmitted frame nothing.
    fn encode(&mut self, pcm: &[i16], output: &mut Vec<u8>) -> usize {
        let mut consumed = 0;
        for frame in pcm.chunks_exact(FRAME_SAMPLES) {
            match self.inner.encode_frame(frame) {
                Encoded::Speech(prm) => pack(&prm, output),
                Encoded::Sid(prm) => pack_sid(prm, output),
                Encoded::NotSent => {}
            }
            consumed += FRAME_SAMPLES;
        }
        consumed
    }

    /// RFC 3551 packetization for DTX: speech frames share a packet, a SID
    /// frame ends one (it must be last), and an untransmitted frame splits
    /// the audio, so the packet after it starts at its own timestamp.
    fn encode_packets(&mut self, pcm: &[i16], packets: &mut Vec<EncodedPacket>) {
        let mut open: Option<EncodedPacket> = None;
        for (k, frame) in pcm.chunks_exact(FRAME_SAMPLES).enumerate() {
            let offset = (k * FRAME_SAMPLES) as u32;
            let encoded = self.inner.encode_frame(frame);
            if encoded == Encoded::NotSent {
                packets.extend(open.take());
                continue;
            }
            let packet = open.get_or_insert_with(|| EncodedPacket {
                offset,
                duration: 0,
                payload: Vec::new(),
            });
            packet.duration += FRAME_SAMPLES as u32;
            match encoded {
                Encoded::Speech(prm) => pack(&prm, &mut packet.payload),
                Encoded::Sid(prm) => {
                    pack_sid(prm, &mut packet.payload);
                    packets.extend(open.take());
                }
                Encoded::NotSent => unreachable!("handled above"),
            }
        }
        packets.extend(open);
    }

    fn payload_type(&self) -> u8 {
        18
    }

    fn codec_type(&self) -> CodecType {
        CodecType::G729
    }
}

/// G.729 Annex A decoder (with the standard postfilter).
pub struct G729Decoder {
    inner: Box<decoder::Decoder>,
}

impl G729Decoder {
    /// Create a new decoder.
    pub fn new() -> Self {
        Self {
            inner: Box::new(decoder::Decoder::new()),
        }
    }

    /// A frame that did not arrive: erasure concealment during speech, the
    /// comfort noise continuing during DTX silence (where a missing frame
    /// is the sender's choice, not a loss).
    fn missing_frame(&mut self, output: &mut Vec<i16>) {
        let out = if self.inner.in_speech() {
            self.inner.decode_frame(&Frame::NotSent, true)
        } else {
            self.inner.decode_frame(&Frame::NotSent, false)
        };
        output.extend_from_slice(&out);
    }

    /// A frame known to have been sent but lost. The ITU test vectors mark
    /// these; over RTP they look like any other missing frame.
    #[cfg(test)]
    fn erase_frame(&mut self, output: &mut Vec<i16>) {
        output.extend_from_slice(&self.inner.decode_frame(&Frame::NotSent, true));
    }

    /// A frame known to have been deliberately not sent. Differs from
    /// [`Self::missing_frame`] only straight after speech, where a DTX
    /// sender would have sent a SID, so over RTP a gap there is a loss.
    #[cfg(test)]
    fn not_sent_frame(&mut self, output: &mut Vec<i16>) {
        output.extend_from_slice(&self.inner.decode_frame(&Frame::NotSent, false));
    }

    #[cfg(test)]
    fn in_speech(&self) -> bool {
        self.inner.in_speech()
    }
}

impl Default for G729Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioDecoder for G729Decoder {
    /// Decode every 10-byte speech frame in the payload, then a trailing
    /// 2-byte Annex B SID frame if there is one. Any other remainder is
    /// ignored.
    fn decode(&mut self, encoded: &[u8], output: &mut Vec<i16>) {
        let frames = encoded.chunks_exact(FRAME_BYTES);
        let rest = frames.remainder();
        for frame in frames {
            let out = self
                .inner
                .decode_frame(&Frame::Speech(unpack(frame)), false);
            output.extend_from_slice(&out);
        }
        if rest.len() == SID_BYTES {
            let out = self
                .inner
                .decode_frame(&Frame::Sid(unpack_sid(rest)), false);
            output.extend_from_slice(&out);
        }
    }

    /// Fill each missing 10 ms frame: erasure concealment after speech,
    /// comfort noise during DTX silence.
    fn conceal_loss(&mut self, samples: usize, _next_packet: Option<&[u8]>, output: &mut Vec<i16>) {
        let target = output.len() + samples;
        while output.len() < target {
            self.missing_frame(output);
        }
        output.truncate(target);
    }

    /// During comfort noise, continue it; after speech, nothing (a stall
    /// that is not DTX plays as silence).
    fn comfort_noise(&mut self, samples: usize, output: &mut Vec<i16>) {
        if self.inner.in_speech() {
            return;
        }
        let target = output.len() + samples;
        while output.len() < target {
            output.extend_from_slice(&self.inner.decode_frame(&Frame::NotSent, false));
        }
        output.truncate(target);
    }

    fn codec_type(&self) -> CodecType {
        CodecType::G729
    }
}

#[cfg(test)]
mod api_tests {
    use super::*;

    fn sine(len: usize) -> Vec<i16> {
        (0..len)
            // 1 kHz: inside the band the codec keeps (it high-passes at 140 Hz).
            .map(|i| ((i as f64 * std::f64::consts::TAU * 1000.0 / 8000.0).sin() * 8000.0) as i16)
            .collect()
    }

    #[test]
    fn pack_and_unpack_roundtrip() {
        let prm: [i16; PRM_SIZE] = [
            0xab, 0x2f1, 0x9c, 1, 0x1abc, 0xa, 0x55, 0x13, 0x0f0f, 0x6, 0x7f,
        ];
        let mut bytes = Vec::new();
        pack(&prm, &mut bytes);
        assert_eq!(bytes.len(), FRAME_BYTES);
        assert_eq!(unpack(&bytes), prm);
    }

    #[test]
    fn packs_msb_first_in_parameter_order() {
        // Only the top bit of L0 (the first parameter's MSB) set.
        let mut prm = [0i16; PRM_SIZE];
        prm[0] = 0x80;
        let mut bytes = Vec::new();
        pack(&prm, &mut bytes);
        assert_eq!(bytes, [0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn encodes_whole_frames_only() {
        let mut enc = G729Encoder::new();
        let mut out = Vec::new();
        assert_eq!(
            enc.encode(&sine(FRAME_SAMPLES * 2 + 30), &mut out),
            FRAME_SAMPLES * 2
        );
        assert_eq!(out.len(), FRAME_BYTES * 2);
        assert_eq!(enc.payload_type(), 18);
        assert_eq!(enc.codec_type(), CodecType::G729);
    }

    #[test]
    fn roundtrip_preserves_a_tone() {
        let input = sine(FRAME_SAMPLES * 50);
        let mut enc = G729Encoder::new();
        let mut dec = G729Decoder::new();
        let mut bits = Vec::new();
        enc.encode(&input, &mut bits);
        let mut out = Vec::new();
        dec.decode(&bits, &mut out);
        assert_eq!(out.len(), input.len());

        // Skip the start-up frames, then compare energies: a CELP codec
        // does not preserve the waveform sample for sample.
        let energy = |x: &[i16]| x.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>();
        let skip = FRAME_SAMPLES * 10;
        let ratio = energy(&out[skip..]) / energy(&input[skip..]);
        assert!((0.5..2.0).contains(&ratio), "energy ratio {ratio}");
    }

    #[test]
    fn concealment_fills_the_requested_length() {
        let mut dec = G729Decoder::new();
        let mut out = Vec::new();
        dec.conceal_loss(160, None, &mut out);
        assert_eq!(out.len(), 160);
        dec.conceal_loss(50, None, &mut out);
        assert_eq!(out.len(), 210);
    }

    #[test]
    fn sid_frame_decodes_to_one_frame() {
        let mut dec = G729Decoder::new();
        let mut out = Vec::new();
        dec.decode(&[0u8; FRAME_BYTES + SID_BYTES], &mut out);
        assert_eq!(out.len(), FRAME_SAMPLES * 2);
    }

    #[test]
    fn sid_pack_and_unpack_roundtrip() {
        let prm = [1, 0x1b, 0x9, 0x15];
        let mut bytes = Vec::new();
        pack_sid(prm, &mut bytes);
        assert_eq!(bytes.len(), SID_BYTES);
        assert_eq!(bytes[1] & 1, 0, "pad bit");
        assert_eq!(unpack_sid(&bytes), prm);
    }

    #[test]
    fn dtx_goes_quiet_in_silence_and_speaks_again() {
        let mut enc = G729Encoder::with_dtx();
        let mut sizes = Vec::new();
        let tone = sine(FRAME_SAMPLES * 100);
        let quiet = vec![0i16; FRAME_SAMPLES * 100];
        for frame in tone
            .chunks(FRAME_SAMPLES)
            .chain(quiet.chunks(FRAME_SAMPLES))
            .chain(tone.chunks(FRAME_SAMPLES))
        {
            let mut out = Vec::new();
            enc.encode(frame, &mut out);
            sizes.push(out.len());
        }
        assert!(
            sizes[..100].iter().all(|&n| n == FRAME_BYTES),
            "tone is speech"
        );
        let silence = &sizes[110..200];
        assert!(
            silence.iter().all(|&n| n == 0 || n == SID_BYTES),
            "silence is SID or nothing"
        );
        assert!(
            silence.iter().filter(|&&n| n == 0).count() > 80,
            "mostly nothing sent"
        );
        assert!(
            sizes[210..].iter().all(|&n| n == FRAME_BYTES),
            "speech resumes"
        );
    }

    /// Every packet `encode_packets` makes is valid RFC 3551 G.729: whole
    /// speech frames, then at most one SID, inside the PCM it came from.
    #[test]
    fn dtx_packets_follow_rfc_3551() {
        let mut enc = G729Encoder::with_dtx();
        let tone = sine(FRAME_SAMPLES * 60);
        let quiet = vec![0i16; FRAME_SAMPLES * 61];
        let pcm: Vec<i16> = tone
            .iter()
            .chain(&quiet)
            .chain(&tone)
            .chain(&quiet)
            .copied()
            .collect();
        let (mut speech, mut sids, mut gaps) = (0, 0, 0);
        for chunk in pcm.chunks_exact(FRAME_SAMPLES * 2) {
            let mut packets = Vec::new();
            enc.encode_packets(chunk, &mut packets);
            let mut end = 0;
            for p in &packets {
                assert!(p.offset >= end, "packets overlap");
                end = p.offset + p.duration;
                assert!(end <= (FRAME_SAMPLES * 2) as u32);
                let frames = p.duration as usize / FRAME_SAMPLES;
                let sid = p.payload.len() % FRAME_BYTES;
                assert!(sid == 0 || sid == SID_BYTES, "payload {}", p.payload.len());
                assert_eq!(
                    p.payload.len() / FRAME_BYTES + usize::from(sid != 0),
                    frames
                );
                speech += p.payload.len() / FRAME_BYTES;
                sids += usize::from(sid != 0);
            }
            gaps += 2 - packets
                .iter()
                .map(|p| p.duration as usize / FRAME_SAMPLES)
                .sum::<usize>();
        }
        assert!(
            speech >= 100 && sids >= 2 && gaps >= 80,
            "{speech} {sids} {gaps}"
        );
    }

    #[test]
    fn default_packetization_is_one_packet() {
        let mut enc = G729Encoder::new();
        let mut packets = Vec::new();
        enc.encode_packets(&vec![0i16; FRAME_SAMPLES * 2], &mut packets);
        assert_eq!(packets.len(), 1);
        assert_eq!((packets[0].offset, packets[0].duration), (0, 160));
        assert_eq!(packets[0].payload.len(), FRAME_BYTES * 2);
    }

    #[test]
    fn comfort_noise_only_after_a_sid() {
        let mut dec = G729Decoder::new();
        let mut out = Vec::new();
        dec.comfort_noise(160, &mut out);
        assert!(out.is_empty(), "no comfort noise during speech");
        dec.decode(&[0x80, 0x55], &mut out);
        out.clear();
        dec.comfort_noise(160, &mut out);
        assert_eq!(out.len(), 160);
    }

    #[test]
    fn dtx_off_never_skips_a_frame() {
        let mut enc = G729Encoder::new();
        let mut out = Vec::new();
        enc.encode(&vec![0i16; FRAME_SAMPLES * 50], &mut out);
        assert_eq!(out.len(), FRAME_BYTES * 50);
    }

    #[test]
    fn missing_frames_in_silence_are_comfort_noise() {
        // A peer's DTX: a SID, then nothing. The gap must sound like the
        // noise the SID described, not decay to silence like an erasure.
        let mut enc = G729Encoder::with_dtx();
        let mut dec = G729Decoder::new();
        let mut x: u32 = 99;
        let noise: Vec<i16> = (0..FRAME_SAMPLES * 300)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ((x % 600) as i16) - 300
            })
            .collect();
        let mut out = Vec::new();
        for frame in noise.chunks(FRAME_SAMPLES) {
            let mut bits = Vec::new();
            enc.encode(frame, &mut bits);
            if bits.is_empty() {
                dec.conceal_loss(FRAME_SAMPLES, None, &mut out);
            } else {
                dec.decode(&bits, &mut out);
            }
        }
        let tail = &out[out.len() - FRAME_SAMPLES * 50..];
        let rms =
            (tail.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(rms > 20.0, "comfort noise rms {rms}");
    }

    #[test]
    fn garbage_input_does_not_panic() {
        let mut dec = G729Decoder::new();
        let mut out = Vec::new();
        let mut x: u32 = 0x1234_5678;
        let junk: Vec<u8> = (0..FRAME_BYTES * 500)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        dec.decode(&junk, &mut out);
        assert_eq!(out.len(), FRAME_SAMPLES * 500);
    }
}

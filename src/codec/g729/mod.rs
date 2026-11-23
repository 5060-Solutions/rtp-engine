//! ITU-T G.729 Annex A (CS-ACELP, 8 kbit/s), in pure Rust.
//!
//! A port of the ITU-T G.729A fixed-point reference implementation
//! (version 1.1, ITU-T G.729 software package release 3), bit-exact with
//! it: the tests in `tests.rs` run the ITU-T test vectors through both the
//! encoder and the decoder. G.729A's bitstream is the same as full G.729's,
//! so it interoperates with any G.729 or G.729A peer.
//!
//! # Framing
//!
//! - **Sample rate**: 8000 Hz
//! - **Frame size**: 10 ms (80 samples), 10 bytes
//! - **Bitrate**: 8 kbit/s
//! - **RTP payload type**: 18, with the 80 bits of each frame packed MSB
//!   first (RFC 3551 §4.5.6). A packet may carry several frames.
//!
//! Annex B (VAD, DTX and comfort noise) is not implemented: the encoder
//! always sends full frames, so offer `annexb=no`. A 2-byte Annex B SID
//! frame from a peer that ignored that is concealed like a lost frame.
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
mod encoder;
mod gain;
mod lpc;
mod lsp;
mod pitch;
mod postfilter;
mod tables;
#[cfg(test)]
mod tests;

use super::{AudioDecoder, AudioEncoder, CodecType};
use tables::BITSNO;

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

/// G.729 Annex A encoder.
pub struct G729Encoder {
    inner: Box<encoder::Encoder>,
}

impl G729Encoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        Self {
            inner: Box::new(encoder::Encoder::new()),
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
    /// left unconsumed.
    fn encode(&mut self, pcm: &[i16], output: &mut Vec<u8>) -> usize {
        let mut consumed = 0;
        for frame in pcm.chunks_exact(FRAME_SAMPLES) {
            let prm = self.inner.encode_frame(frame);
            pack(&prm, output);
            consumed += FRAME_SAMPLES;
        }
        consumed
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

    fn conceal_frame(&mut self, output: &mut Vec<i16>) {
        let frame = self.inner.decode_frame(&[0; PRM_SIZE], true);
        output.extend_from_slice(&frame);
    }
}

impl Default for G729Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioDecoder for G729Decoder {
    /// Decode every 10-byte frame in the payload. A trailing 2-byte Annex B
    /// SID frame is concealed; any other remainder is ignored.
    fn decode(&mut self, encoded: &[u8], output: &mut Vec<i16>) {
        let frames = encoded.chunks_exact(FRAME_BYTES);
        let rest = frames.remainder().len();
        for frame in frames {
            let out = self.inner.decode_frame(&unpack(frame), false);
            output.extend_from_slice(&out);
        }
        if rest == SID_BYTES {
            self.conceal_frame(output);
        }
    }

    /// Run G.729's frame-erasure concealment for each lost 10 ms frame.
    fn conceal_loss(&mut self, samples: usize, _next_packet: Option<&[u8]>, output: &mut Vec<i16>) {
        let target = output.len() + samples;
        while output.len() < target {
            self.conceal_frame(output);
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
    fn sid_frame_is_concealed_not_dropped() {
        let mut dec = G729Decoder::new();
        let mut out = Vec::new();
        dec.decode(&[0u8; FRAME_BYTES + SID_BYTES], &mut out);
        assert_eq!(out.len(), FRAME_SAMPLES * 2);
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

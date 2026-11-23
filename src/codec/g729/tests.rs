//! Bit-exactness against the ITU-T G.729 Annex A test vectors.
//!
//! The vectors ship in the ITU-T G.729 software package and are not
//! committed here. `scripts/fetch-g729-vectors.sh` downloads and unpacks
//! them into `target/g729-vectors`; set `G729_TEST_VECTORS` to use another
//! directory. Without them these tests say so and pass, unless
//! `G729_REQUIRE_VECTORS` is set, as CI sets it.

use std::path::PathBuf;

use super::{FRAME_BYTES, FRAME_SAMPLES, G729Decoder, G729Encoder};
use crate::codec::{AudioDecoder, AudioEncoder};

/// ITU serial-format words: a sync word, a size word, then one word per bit.
const SERIAL_WORDS: usize = 82;
const BIT_1: i16 = 0x0081;

fn vector_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("G729_TEST_VECTORS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/g729-vectors"),
        PathBuf::from,
    );
    if dir.join("SPEECH.BIT").is_file() {
        return Some(dir);
    }
    let msg = format!(
        "G.729 test vectors not found in {}; run scripts/fetch-g729-vectors.sh",
        dir.display()
    );
    assert!(std::env::var_os("G729_REQUIRE_VECTORS").is_none(), "{msg}");
    eprintln!("{msg}");
    None
}

fn read_words(dir: &std::path::Path, name: &str) -> Vec<i16> {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// One ITU serial frame as the RTP payload (RFC 3551: the same 80 bits,
/// packed MSB first). `None` for an erased frame, which the ITU format marks
/// with zero words in place of bits.
fn serial_to_payload(frame: &[i16]) -> Option<[u8; FRAME_BYTES]> {
    let bits = &frame[2..];
    if bits.contains(&0) {
        return None;
    }
    let mut out = [0u8; FRAME_BYTES];
    for (i, &b) in bits.iter().enumerate() {
        if b == BIT_1 {
            out[i / 8] |= 0x80 >> (i % 8);
        }
    }
    Some(out)
}

fn check_encoder(name: &str) {
    let Some(dir) = vector_dir() else { return };
    let pcm = read_words(&dir, &format!("{name}.IN"));
    let serial = read_words(&dir, &format!("{name}.BIT"));

    let mut enc = G729Encoder::new();
    for (n, (input, expected)) in pcm
        .chunks_exact(FRAME_SAMPLES)
        .zip(serial.chunks_exact(SERIAL_WORDS))
        .enumerate()
    {
        let mut out = Vec::new();
        assert_eq!(enc.encode(input, &mut out), FRAME_SAMPLES);
        let expected = serial_to_payload(expected).expect("encoder vectors have no erasures");
        assert_eq!(out, expected, "{name}: frame {n} differs");
    }
}

fn check_decoder(name: &str, pst: &str) {
    let Some(dir) = vector_dir() else { return };
    let serial = read_words(&dir, &format!("{name}.BIT"));
    let expected = read_words(&dir, pst);

    let mut dec = G729Decoder::new();
    let mut out = Vec::new();
    for frame in serial.chunks_exact(SERIAL_WORDS) {
        match serial_to_payload(frame) {
            Some(payload) => dec.decode(&payload, &mut out),
            None => dec.conceal_loss(FRAME_SAMPLES, None, &mut out),
        }
    }
    assert_eq!(out.len(), expected.len(), "{name}: length");
    if let Some(i) = out.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{name}: sample {i} (frame {}) is {}, expected {}",
            i / FRAME_SAMPLES,
            out[i],
            expected[i]
        );
    }
}

macro_rules! vector_tests {
    ($($enc:ident / $dec:ident: $name:literal, $pst:literal;)*) => {$(
        #[test]
        fn $enc() {
            check_encoder($name);
        }

        #[test]
        fn $dec() {
            check_decoder($name, $pst);
        }
    )*};
}

vector_tests! {
    encodes_algthm / decodes_algthm: "ALGTHM", "ALGTHM.PST";
    encodes_fixed / decodes_fixed: "FIXED", "FIXED.PST";
    encodes_lsp / decodes_lsp: "LSP", "LSP.PST";
    encodes_pitch / decodes_pitch: "PITCH", "PITCH.PST";
    encodes_speech / decodes_speech: "SPEECH", "SPEECH.PST";
    encodes_tame / decodes_tame: "TAME", "TAME.PST";
    encodes_test / decodes_test: "TEST", "TEST.pst";
}

#[test]
fn decodes_erasure() {
    check_decoder("ERASURE", "ERASURE.PST");
}

#[test]
fn decodes_overflow() {
    check_decoder("OVERFLOW", "OVERFLOW.PST");
}

#[test]
fn decodes_parity() {
    check_decoder("PARITY", "PARITY.PST");
}

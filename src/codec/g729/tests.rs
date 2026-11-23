//! Bit-exactness against the ITU-T G.729 Annex A and Annex B test vectors.
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
    if dir.join("SPEECH.BIT").is_file() && dir.join("tstseq1a.bit").is_file() {
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

// Annex B (VAD/DTX/CNG). These bitstreams mix frame sizes: 80 bits for
// speech, 16 for a SID (15 bits and a zero pad, as RFC 3551 carries it in
// two bytes) and 0 for a frame the encoder did not transmit.

/// One Annex B frame, as the RTP layer sees it.
enum Received {
    /// A speech or SID payload.
    Payload(Vec<u8>),
    /// Nothing was sent (DTX).
    NotSent,
    /// Sent but lost: the ITU format marks it with zero bits, or for an
    /// untransmitted frame with a corrupted sync word.
    Erased,
}

fn annex_b_frames(words: &[i16]) -> Vec<Received> {
    let mut frames = Vec::new();
    let mut i = 0;
    while i + 2 <= words.len() {
        let (sync, size) = (words[i], words[i + 1] as usize);
        let bits = &words[i + 2..i + 2 + size];
        i += 2 + size;
        let erased = if size == 0 {
            sync != 0x6b21
        } else {
            bits.contains(&0)
        };
        frames.push(if erased {
            Received::Erased
        } else if size == 0 {
            Received::NotSent
        } else {
            let mut out = vec![0u8; size / 8];
            for (k, &b) in bits.iter().enumerate() {
                if b == BIT_1 {
                    out[k / 8] |= 0x80 >> (k % 8);
                }
            }
            Received::Payload(out)
        });
    }
    frames
}

fn check_annex_b_encoder(input: &str, bitstream: &str) {
    let Some(dir) = vector_dir() else { return };
    let pcm = read_words(&dir, input);
    let expected = annex_b_frames(&read_words(&dir, bitstream));

    let mut enc = G729Encoder::with_dtx();
    for (n, (frame, expected)) in pcm.chunks_exact(FRAME_SAMPLES).zip(&expected).enumerate() {
        let mut out = Vec::new();
        assert_eq!(enc.encode(frame, &mut out), FRAME_SAMPLES);
        let expected: &[u8] = match expected {
            Received::Payload(p) => p,
            Received::NotSent => &[],
            Received::Erased => panic!("{bitstream}: encoder output has no erasures"),
        };
        assert_eq!(out, expected, "{bitstream}: frame {n} differs");
    }
    assert_eq!(
        pcm.len() / FRAME_SAMPLES,
        expected.len(),
        "{bitstream}: frame count"
    );
}

fn check_annex_b_decoder(bitstream: &str, output: &str) {
    let Some(dir) = vector_dir() else { return };
    let frames = annex_b_frames(&read_words(&dir, bitstream));
    let expected = read_words(&dir, output);

    let mut dec = G729Decoder::new();
    let mut out = Vec::new();
    for frame in &frames {
        match frame {
            Received::Payload(p) => dec.decode(p, &mut out),
            // Over RTP an untransmitted frame is a missing packet. Straight
            // after speech that reads as a loss (a DTX sender opens every
            // silence with a SID), so only there does the test say
            // explicitly that the frame was not sent.
            Received::NotSent if dec.in_speech() => dec.not_sent_frame(&mut out),
            Received::NotSent => dec.conceal_loss(FRAME_SAMPLES, None, &mut out),
            Received::Erased => dec.erase_frame(&mut out),
        }
    }
    assert_eq!(out.len(), expected.len(), "{bitstream}: length");
    if let Some(i) = out.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{bitstream}: sample {i} (frame {}) is {}, expected {}",
            i / FRAME_SAMPLES,
            out[i],
            expected[i]
        );
    }
}

macro_rules! annex_b_tests {
    ($($enc:ident / $dec:ident: $n:literal;)*) => {$(
        #[test]
        fn $enc() {
            check_annex_b_encoder(concat!("tstseq", $n, ".bin"), concat!("tstseq", $n, "a.bit"));
        }

        #[test]
        fn $dec() {
            check_annex_b_decoder(concat!("tstseq", $n, "a.bit"), concat!("tstseq", $n, "a.out"));
        }
    )*};
}

annex_b_tests! {
    annex_b_encodes_tstseq1 / annex_b_decodes_tstseq1: "1";
    annex_b_encodes_tstseq2 / annex_b_decodes_tstseq2: "2";
    annex_b_encodes_tstseq3 / annex_b_decodes_tstseq3: "3";
    annex_b_encodes_tstseq4 / annex_b_decodes_tstseq4: "4";
}

#[test]
fn annex_b_decodes_tstseq5() {
    check_annex_b_decoder("tstseq5.bit", "tstseq5a.out");
}

/// Includes an erased SID and an erased untransmitted frame.
#[test]
fn annex_b_decodes_tstseq6() {
    check_annex_b_decoder("tstseq6.bit", "tstseq6a.out");
}

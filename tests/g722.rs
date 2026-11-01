//! G.722 against an independent implementation, plus its RTP framing.

use rtp_engine::codec::{AudioDecoder, AudioEncoder, negotiate_codec_with_pt};
use rtp_engine::codec::{CodecType, G722Decoder, G722Encoder, create_decoder, create_encoder};

fn samples(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

const INPUT: &[u8] = include_bytes!("fixtures/g722/input.s16le");
const FFMPEG_CODED: &[u8] = include_bytes!("fixtures/g722/ffmpeg.g722");
const FFMPEG_DECODED: &[u8] = include_bytes!("fixtures/g722/ffmpeg_decoded.s16le");

#[test]
fn g722_encoder_is_bit_exact_with_ffmpeg() {
    let mut encoder = G722Encoder::new();
    let mut coded = Vec::new();
    // In 20 ms packets, as a call sends it.
    for packet in samples(INPUT).chunks(320) {
        encoder.encode(packet, &mut coded);
    }
    assert_eq!(coded.len(), FFMPEG_CODED.len());
    let first_difference = coded.iter().zip(FFMPEG_CODED).position(|(a, b)| a != b);
    assert_eq!(
        first_difference, None,
        "encoder output differs from ffmpeg's"
    );
}

#[test]
fn g722_decoder_is_bit_exact_with_ffmpeg() {
    let mut decoder = G722Decoder::new();
    let mut pcm = Vec::new();
    for packet in FFMPEG_CODED.chunks(160) {
        decoder.decode(packet, &mut pcm);
    }
    let expected = samples(FFMPEG_DECODED);
    assert_eq!(pcm.len(), expected.len());
    let first_difference = pcm.iter().zip(&expected).position(|(a, b)| a != b);
    assert_eq!(
        first_difference, None,
        "decoder output differs from ffmpeg's"
    );
}

/// A wideband tone survives the round trip with a good signal-to-noise
/// ratio, once the codec's QMF delay is allowed for.
#[test]
fn g722_encode_decode_snr() {
    let tone: Vec<i16> = (0..16_000)
        .map(|n| {
            let t = f64::from(n) / 16_000.0;
            (10_000.0 * (2.0 * std::f64::consts::PI * 1_000.0 * t).sin()) as i16
        })
        .collect();
    let mut encoder = create_encoder(CodecType::G722).unwrap();
    let mut decoder = create_decoder(CodecType::G722).unwrap();
    let mut coded = Vec::new();
    encoder.encode(&tone, &mut coded);
    assert_eq!(coded.len(), 8_000, "one byte per two samples");
    let mut out = Vec::new();
    decoder.decode(&coded, &mut out);
    assert_eq!(out.len(), tone.len());

    // Skip the first 100 ms while the predictors adapt, and find the delay.
    let snr = |delay: usize| {
        let (mut signal, mut noise) = (0.0_f64, 0.0_f64);
        for n in 1_600..tone.len() - 64 {
            let s = f64::from(tone[n]);
            let e = f64::from(out[n + delay]) - s;
            signal += s * s;
            noise += e * e;
        }
        10.0 * (signal / noise.max(1.0)).log10()
    };
    let best = (0..64).map(snr).fold(f64::MIN, f64::max);
    assert!(best > 25.0, "SNR {best:.1} dB");
}

/// RFC 3551 §4.5.2: G.722 carries 16 kHz audio on an 8000 Hz RTP clock.
#[test]
fn g722_rtp_clock_is_8000() {
    let codec = CodecType::G722;
    assert_eq!(codec.payload_type(), 9);
    assert_eq!(codec.clock_rate(), 8000);
    assert_eq!(codec.pcm_rate(), 16_000);
    assert_eq!(codec.pcm_samples_per_packet(), 320);
    assert_eq!(codec.rtp_ticks_per_packet(), 160);
    assert_eq!(codec.name(), "G722");
    assert_eq!(CodecType::from_payload_type(9), Some(codec));
    assert_eq!(CodecType::from_name("g722"), Some(codec));
    assert!(CodecType::all().contains(&codec));
}

#[test]
fn g722_negotiates_from_sdp() {
    let sdp = "m=audio 4000 RTP/AVP 9 0\r\na=rtpmap:9 G722/8000\r\n";
    assert_eq!(negotiate_codec_with_pt(sdp), Some((CodecType::G722, 9)));
    let static_only = "m=audio 4000 RTP/AVP 9\r\n";
    assert_eq!(
        negotiate_codec_with_pt(static_only),
        Some((CodecType::G722, 9))
    );
}

/// An odd number of samples is held for the next call, not dropped.
#[test]
fn g722_encoder_consumes_odd_lengths() {
    let mut encoder = G722Encoder::new();
    let mut out = Vec::new();
    assert_eq!(encoder.encode(&[1, 2, 3], &mut out), 3);
    assert_eq!(out.len(), 1);
    assert_eq!(encoder.encode(&[4], &mut out), 1);
    assert_eq!(out.len(), 2);
    assert_eq!(encoder.codec_type(), CodecType::G722);
}

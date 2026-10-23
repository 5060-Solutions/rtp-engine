//! Opus codec implementation.
//!
//! Opus runs at 48 kHz, and so does this codec: the encoder takes 48 kHz PCM
//! and the decoder produces it. The media path resamples between the device
//! rate and 48 kHz, so Opus carries full-band audio rather than being
//! squeezed through the 8 kHz the telephone codecs use.
//!
//! The encoder is tuned for VoIP: a fixed 32 kbit/s bitrate and in-band FEC,
//! which puts a low-rate copy of each frame into the following packet so a
//! single lost packet can be reconstructed by [`OpusDecoder::conceal_loss`].

use super::{AudioDecoder, AudioEncoder, CodecType};
use crate::error::{Error, Result};

/// The sample rate Opus works at, which is also its RTP clock.
const OPUS_RATE: u32 = 48_000;

/// Encoder bitrate. 32 kbit/s is comfortably transparent for mono speech at
/// full band and leaves room for the FEC copy.
const BITRATE_BPS: i32 = 32_000;

/// The loss the encoder plans for. libopus only emits FEC data when it is
/// told to expect some loss; 10 % is enough to turn it on without the
/// encoder sacrificing much of the primary frame's quality.
const EXPECTED_LOSS_PERCENT: u8 = 10;

/// Largest frame libopus can return for one packet: 120 ms at 48 kHz.
const MAX_DECODED_SAMPLES: usize = 5760;

/// Frame lengths libopus accepts at 48 kHz: 2.5, 5, 10, 20, 40 and 60 ms.
const VALID_FRAME_LENGTHS: [usize; 6] = [120, 240, 480, 960, 1920, 2880];

/// Opus encoder: 48 kHz mono in, Opus packets out.
pub struct OpusEncoder {
    encoder: audiopus::coder::Encoder,
}

impl OpusEncoder {
    /// Create a new Opus encoder.
    pub fn new() -> Result<Self> {
        let mut encoder = audiopus::coder::Encoder::new(
            audiopus::SampleRate::Hz48000,
            audiopus::Channels::Mono,
            audiopus::Application::Voip,
        )
        .map_err(|e| Error::codec(format!("Opus encoder init: {}", e)))?;

        encoder
            .set_bitrate(audiopus::Bitrate::BitsPerSecond(BITRATE_BPS))
            .map_err(|e| Error::codec(format!("Opus set bitrate: {}", e)))?;
        encoder
            .set_signal(audiopus::Signal::Voice)
            .map_err(|e| Error::codec(format!("Opus set signal: {}", e)))?;
        encoder
            .set_inband_fec(true)
            .map_err(|e| Error::codec(format!("Opus set FEC: {}", e)))?;
        encoder
            .set_packet_loss_perc(EXPECTED_LOSS_PERCENT)
            .map_err(|e| Error::codec(format!("Opus set loss: {}", e)))?;

        Ok(Self { encoder })
    }
}

impl AudioEncoder for OpusEncoder {
    /// Encode one Opus frame of 48 kHz PCM.
    ///
    /// `pcm` must be exactly one of the frame lengths Opus accepts (960 for
    /// the 20 ms packets the session sends). Any other length is refused —
    /// nothing is written and 0 is returned — because libopus would reject
    /// it and the alternative, encoding a prefix and dropping the rest,
    /// desynchronises the RTP timestamp from the audio.
    fn encode(&mut self, pcm: &[i16], output: &mut Vec<u8>) -> usize {
        if !VALID_FRAME_LENGTHS.contains(&pcm.len()) {
            if !pcm.is_empty() {
                log::warn!(
                    "Opus encode: {} samples is not a valid 48 kHz frame length, dropped",
                    pcm.len()
                );
            }
            return 0;
        }

        let mut encoded = [0u8; 4000];
        match self.encoder.encode(pcm, &mut encoded) {
            Ok(len) => {
                output.extend_from_slice(&encoded[..len]);
                pcm.len()
            }
            Err(e) => {
                log::error!("Opus encode error: {}", e);
                0
            }
        }
    }

    fn payload_type(&self) -> u8 {
        111
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Opus
    }
}

impl std::fmt::Debug for OpusEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusEncoder")
            .field("rate", &OPUS_RATE)
            .field("bitrate_bps", &BITRATE_BPS)
            .finish()
    }
}

/// Opus decoder: Opus packets in, 48 kHz mono PCM out.
pub struct OpusDecoder {
    decoder: audiopus::coder::Decoder,
}

impl OpusDecoder {
    /// Create a new Opus decoder.
    pub fn new() -> Result<Self> {
        let decoder =
            audiopus::coder::Decoder::new(audiopus::SampleRate::Hz48000, audiopus::Channels::Mono)
                .map_err(|e| Error::codec(format!("Opus decoder init: {}", e)))?;

        Ok(Self { decoder })
    }

    /// Run the decoder once and append what it produced.
    ///
    /// `packet` is `None` for packet-loss concealment, in which case `frame`
    /// is how many samples to synthesise. With `fec` set, `packet` is the
    /// packet *after* the lost one and libopus decodes the redundant copy of
    /// the lost frame that it carries, falling back to PLC if there is none.
    fn run(&mut self, packet: Option<&[u8]>, frame: usize, fec: bool, output: &mut Vec<i16>) {
        let mut decoded = [0i16; MAX_DECODED_SAMPLES];
        let frame = frame.min(MAX_DECODED_SAMPLES);

        let packet: Option<audiopus::packet::Packet<'_>> = match packet {
            Some(bytes) => match bytes.try_into() {
                Ok(p) => Some(p),
                Err(e) => {
                    log::error!("Opus packet error: {}", e);
                    return;
                }
            },
            None => None,
        };
        let out_signals: audiopus::MutSignals<'_, _> = match (&mut decoded[..frame]).try_into() {
            Ok(s) => s,
            Err(e) => {
                log::error!("Opus signals error: {}", e);
                return;
            }
        };

        match self.decoder.decode(packet, out_signals, fec) {
            Ok(samples) => output.extend_from_slice(&decoded[..samples]),
            Err(e) => log::error!("Opus decode error: {}", e),
        }
    }
}

impl AudioDecoder for OpusDecoder {
    fn decode(&mut self, encoded: &[u8], output: &mut Vec<i16>) {
        if encoded.is_empty() {
            return;
        }
        self.run(Some(encoded), MAX_DECODED_SAMPLES, false, output);
    }

    /// Fill in a lost packet: from the FEC data in the next packet when it
    /// is already here, otherwise from the decoder's own concealment, which
    /// extrapolates the last good frame instead of dropping to silence.
    fn conceal_loss(&mut self, samples: usize, next_packet: Option<&[u8]>, output: &mut Vec<i16>) {
        match next_packet {
            Some(next) if !next.is_empty() => self.run(Some(next), samples, true, output),
            _ => self.run(None, samples, false, output),
        }
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Opus
    }
}

impl std::fmt::Debug for OpusDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusDecoder")
            .field("rate", &OPUS_RATE)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 20 ms frame of a 440 Hz tone at 48 kHz, 960 samples.
    fn tone_frame(offset: usize) -> Vec<i16> {
        (offset..offset + 960)
            .map(|i| {
                let t = i as f64 / f64::from(OPUS_RATE);
                (f64::sin(2.0 * std::f64::consts::PI * 440.0 * t) * 16000.0) as i16
            })
            .collect()
    }

    #[test]
    fn test_opus_encoder_creation() {
        let encoder = OpusEncoder::new();
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_decoder_creation() {
        let decoder = OpusDecoder::new();
        assert!(decoder.is_ok());
    }

    /// A 960-sample frame in gives a 960-sample frame back: no resampling
    /// happens inside the codec any more.
    #[test]
    fn test_opus_roundtrip() {
        let mut encoder = OpusEncoder::new().unwrap();
        let mut decoder = OpusDecoder::new().unwrap();

        let input = tone_frame(0);
        let mut encoded = Vec::new();
        let consumed = encoder.encode(&input, &mut encoded);
        assert_eq!(consumed, 960);
        assert_ne!(encoded, [] as [u8; 0]);

        let mut decoded = Vec::new();
        decoder.decode(&encoded, &mut decoded);
        assert_eq!(decoded.len(), 960);
    }

    /// The codec agrees with `CodecType` on its rate and packet size.
    #[test]
    fn opus_frame_matches_codec_type_packet_size() {
        assert_eq!(CodecType::Opus.pcm_rate(), OPUS_RATE);
        assert_eq!(CodecType::Opus.pcm_samples_per_packet(), 960);
    }

    /// An input that is not a whole Opus frame is refused — nothing written,
    /// nothing consumed, no panic — rather than handed to libopus, which
    /// would reject it, or trimmed, which would skew the timestamp.
    #[test]
    fn encode_refuses_a_non_frame_length_without_panicking() {
        let mut encoder = OpusEncoder::new().unwrap();
        for len in [1usize, 160, 959, 961, 1000] {
            let input = vec![0i16; len];
            let mut encoded = Vec::new();
            let consumed = encoder.encode(&input, &mut encoded);
            assert_eq!(consumed, 0, "{} samples should be refused", len);
            assert!(encoded.is_empty(), "{} samples should write nothing", len);
        }
        // Empty input is simply nothing to do.
        let mut encoded = Vec::new();
        assert_eq!(encoder.encode(&[], &mut encoded), 0);
        assert_eq!(encoded, [] as [u8; 0]);
    }

    /// Packet-loss concealment produces a full frame of audio, and after a
    /// run of real frames it is not silence: the decoder extrapolates.
    #[test]
    fn conceal_loss_without_fec_extrapolates_a_frame() {
        let mut encoder = OpusEncoder::new().unwrap();
        let mut decoder = OpusDecoder::new().unwrap();

        for n in 0..10 {
            let mut encoded = Vec::new();
            encoder.encode(&tone_frame(n * 960), &mut encoded);
            let mut decoded = Vec::new();
            decoder.decode(&encoded, &mut decoded);
        }

        let mut concealed = Vec::new();
        decoder.conceal_loss(960, None, &mut concealed);
        assert_eq!(concealed.len(), 960);
        assert!(
            concealed.iter().any(|&s| s != 0),
            "PLC after a tone should not be silence"
        );
    }

    /// With the next packet in hand, the lost frame is rebuilt from its
    /// in-band FEC copy, and still yields exactly one frame.
    #[test]
    fn conceal_loss_with_fec_recovers_a_frame_from_the_next_packet() {
        let mut encoder = OpusEncoder::new().unwrap();
        let mut decoder = OpusDecoder::new().unwrap();

        let mut packets = Vec::new();
        for n in 0..12 {
            let mut encoded = Vec::new();
            encoder.encode(&tone_frame(n * 960), &mut encoded);
            packets.push(encoded);
        }
        for pkt in &packets[..10] {
            let mut decoded = Vec::new();
            decoder.decode(pkt, &mut decoded);
        }

        // Packet 10 is lost; packet 11 has arrived.
        let mut recovered = Vec::new();
        decoder.conceal_loss(960, Some(&packets[11]), &mut recovered);
        assert_eq!(recovered.len(), 960);
        assert!(recovered.iter().any(|&s| s != 0));

        // The stream then continues with the packet that carried the FEC.
        let mut next = Vec::new();
        decoder.decode(&packets[11], &mut next);
        assert_eq!(next.len(), 960);
    }
}

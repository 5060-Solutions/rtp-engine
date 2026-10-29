//! G.722 wideband codec (ITU-T G.722, 64 kbit/s mode).
//!
//! Sub-band ADPCM: a QMF splits 16 kHz audio into a 0–4 kHz and a 4–8 kHz
//! band at 8 kHz each, the low band is coded with 6 bits per sample and the
//! high band with 2, and each sample pair packs into one byte. So 20 ms is
//! 320 PCM samples in and 160 bytes out.
//!
//! The RTP clock is 8000 even though the audio is 16 kHz — RFC 3551 §4.5.2
//! keeps the value an error in the original spec made — which is why
//! [`super::CodecType::clock_rate`] and [`super::CodecType::pcm_rate`] differ
//! for this codec.
//!
//! Written from the ITU-T G.722 (09/2012) block descriptions, in the
//! integer form of the reference implementation. The tests check it
//! bit-exact against an independent implementation (ffmpeg's).

use super::{AudioDecoder, AudioEncoder, CodecType};

/// The 24-tap transmit and receive QMF, as 12 coefficients used mirrored.
const QMF: [i32; 12] = [3, -11, 12, 32, -210, 951, 3876, -805, 362, -156, 53, -11];

/// Low band quantizer decision levels (QUANTL), for the 6-bit codes.
const Q6: [i32; 32] = [
    0, 35, 72, 110, 150, 190, 233, 276, 323, 370, 422, 473, 530, 587, 650, 714, 786, 858, 940,
    1023, 1121, 1219, 1339, 1458, 1612, 1765, 1980, 2195, 2557, 2919, 0, 0,
];
/// Codes for a negative low band difference, by decision interval.
const ILN: [i32; 32] = [
    0, 63, 62, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
    10, 9, 8, 7, 6, 5, 4, 0,
];
/// Codes for a positive low band difference, by decision interval.
const ILP: [i32; 32] = [
    0, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39,
    38, 37, 36, 35, 34, 33, 32, 0,
];
/// Log scale factor multipliers for the low band (LOGSCL).
const WL: [i32; 8] = [-60, -30, 58, 172, 334, 538, 1198, 3042];
/// Maps a 4-bit low band code to its WL index.
const RL42: [i32; 16] = [0, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 0];
/// The antilog table for scale factor computation (SCALEL / SCALEH).
const ILB: [i32; 32] = [
    2048, 2093, 2139, 2186, 2233, 2282, 2332, 2383, 2435, 2489, 2543, 2599, 2656, 2714, 2774, 2834,
    2896, 2960, 3025, 3091, 3158, 3228, 3298, 3371, 3444, 3520, 3597, 3676, 3756, 3838, 3922, 4008,
];
/// Inverse quantizer for the 4-bit (predictor-adaptation) low band code.
const QM4: [i32; 16] = [
    0, -20456, -12896, -8968, -6288, -4240, -2584, -1200, 20456, 12896, 8968, 6288, 4240, 2584,
    1200, 0,
];
/// Inverse quantizer for the full 6-bit low band code (decoder output).
const QM6: [i32; 64] = [
    -136, -136, -136, -136, -24808, -21904, -19008, -16704, -14984, -13512, -12280, -11192, -10232,
    -9360, -8576, -7856, -7192, -6576, -6000, -5456, -4944, -4464, -4008, -3576, -3168, -2776,
    -2400, -2032, -1688, -1360, -1040, -728, 24808, 21904, 19008, 16704, 14984, 13512, 12280,
    11192, 10232, 9360, 8576, 7856, 7192, 6576, 6000, 5456, 4944, 4464, 4008, 3576, 3168, 2776,
    2400, 2032, 1688, 1360, 1040, 728, 432, 136, -432, -136,
];
/// Inverse quantizer for the 2-bit high band code.
const QM2: [i32; 4] = [-7408, -1616, 7408, 1616];
/// High band codes by quantizer interval, for negative and positive differences.
const IHN: [i32; 3] = [0, 1, 0];
const IHP: [i32; 3] = [0, 3, 2];
/// Log scale factor multipliers for the high band (LOGSCH).
const WH: [i32; 3] = [0, -214, 798];
/// Maps a 2-bit high band code to its WH index.
const RH2: [i32; 4] = [2, 1, 2, 1];

/// Clamp to the 16-bit range, as the reference does after every add.
fn saturate(x: i32) -> i32 {
    x.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}

/// The adaptive predictor and scale factor state of one sub-band.
#[derive(Debug, Clone, Copy)]
struct Band {
    /// Signal estimate.
    s: i32,
    /// Pole section output.
    sp: i32,
    /// Zero section output.
    sz: i32,
    /// Reconstructed signal history.
    r: [i32; 3],
    /// Pole coefficients.
    a: [i32; 3],
    /// Updated pole coefficients.
    ap: [i32; 3],
    /// Partial reconstructed signal history.
    p: [i32; 3],
    /// Quantized difference history.
    d: [i32; 7],
    /// Zero coefficients.
    b: [i32; 7],
    /// Updated zero coefficients.
    bp: [i32; 7],
    /// Log scale factor.
    nb: i32,
    /// Scale factor.
    det: i32,
}

impl Band {
    fn new(det: i32) -> Self {
        Self {
            s: 0,
            sp: 0,
            sz: 0,
            r: [0; 3],
            a: [0; 3],
            ap: [0; 3],
            p: [0; 3],
            d: [0; 7],
            b: [0; 7],
            bp: [0; 7],
            nb: 0,
            det,
        }
    }

    /// Blocks 3L/3H: adapt the log scale factor and recompute the scale.
    fn scale(&mut self, w: i32, max_nb: i32, shift: i32) {
        self.nb = ((self.nb * 127) >> 7) + w;
        self.nb = self.nb.clamp(0, max_nb);
        let wd1 = ((self.nb >> 6) & 31) as usize;
        let wd2 = shift - (self.nb >> 11);
        let wd3 = if wd2 < 0 {
            ILB[wd1] << -wd2
        } else {
            ILB[wd1] >> wd2
        };
        self.det = wd3 << 2;
    }

    /// Block 4: reconstruct, update the predictor, and predict the next
    /// sample, from the quantized difference `d`.
    fn predict(&mut self, d: i32) {
        let sign = |x: i32| x >> 15;

        // RECONS and PARREC.
        self.d[0] = d;
        self.r[0] = saturate(self.s + d);
        self.p[0] = saturate(self.sz + d);

        // UPPOL2.
        let sg = [sign(self.p[0]), sign(self.p[1]), sign(self.p[2])];
        let wd1 = saturate(self.a[1] << 2);
        let wd2 = if sg[0] == sg[1] { -wd1 } else { wd1 }.min(32767);
        let mut wd3 = (wd2 >> 7) + if sg[0] == sg[2] { 128 } else { -128 };
        wd3 += (self.a[2] * 32512) >> 15;
        self.ap[2] = wd3.clamp(-12288, 12288);

        // UPPOL1.
        let wd1 = if sg[0] == sg[1] { 192 } else { -192 };
        let wd2 = (self.a[1] * 32640) >> 15;
        self.ap[1] = saturate(wd1 + wd2);
        let limit = saturate(15360 - self.ap[2]);
        self.ap[1] = self.ap[1].clamp(-limit, limit);

        // UPZERO.
        let step = if d == 0 { 0 } else { 128 };
        let sg0 = sign(d);
        for i in 1..7 {
            let wd2 = if sign(self.d[i]) == sg0 { step } else { -step };
            let wd3 = (self.b[i] * 32640) >> 15;
            self.bp[i] = saturate(wd2 + wd3);
        }

        // DELAYA.
        for i in (1..7).rev() {
            self.d[i] = self.d[i - 1];
            self.b[i] = self.bp[i];
        }
        for i in (1..3).rev() {
            self.r[i] = self.r[i - 1];
            self.p[i] = self.p[i - 1];
            self.a[i] = self.ap[i];
        }

        // FILTEP.
        let wd1 = (self.a[1] * saturate(self.r[1] + self.r[1])) >> 15;
        let wd2 = (self.a[2] * saturate(self.r[2] + self.r[2])) >> 15;
        self.sp = saturate(wd1 + wd2);

        // FILTEZ.
        let mut sz = 0;
        for i in (1..7).rev() {
            sz += (self.b[i] * saturate(self.d[i] + self.d[i])) >> 15;
        }
        self.sz = saturate(sz);

        // PREDIC.
        self.s = saturate(self.sp + self.sz);
    }
}

/// Scale factor adaptation limits and shifts for the two bands.
const LOW_MAX_NB: i32 = 18432;
const HIGH_MAX_NB: i32 = 22528;
const LOW_SHIFT: i32 = 8;
const HIGH_SHIFT: i32 = 10;

/// G.722 encoder: 16 kHz PCM in, one byte per two samples out.
#[derive(Debug, Clone)]
pub struct G722Encoder {
    x: [i32; 24],
    low: Band,
    high: Band,
    /// An odd trailing sample held for the next call.
    pending: Option<i16>,
}

impl Default for G722Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl G722Encoder {
    /// A new encoder in its reset state.
    pub fn new() -> Self {
        Self {
            x: [0; 24],
            low: Band::new(32),
            high: Band::new(8),
            pending: None,
        }
    }

    fn encode_pair(&mut self, first: i16, second: i16) -> u8 {
        // Transmit QMF.
        self.x.copy_within(2.., 0);
        self.x[22] = i32::from(first);
        self.x[23] = i32::from(second);
        let (mut sum_odd, mut sum_even) = (0, 0);
        for i in 0..12 {
            sum_odd += self.x[2 * i] * QMF[i];
            sum_even += self.x[2 * i + 1] * QMF[11 - i];
        }
        let xlow = (sum_even + sum_odd) >> 14;
        let xhigh = (sum_even - sum_odd) >> 14;

        // Low band: SUBTRA, QUANTL, INVQAL, LOGSCL, SCALEL, then predict.
        let el = saturate(xlow - self.low.s);
        let wd = if el >= 0 { el } else { -(el + 1) };
        let mut i = 1;
        while i < 30 && wd >= (Q6[i] * self.low.det) >> 12 {
            i += 1;
        }
        let ilow = if el < 0 { ILN[i] } else { ILP[i] };
        let ril = (ilow >> 2) as usize;
        let dlow = (self.low.det * QM4[ril]) >> 15;
        self.low
            .scale(WL[RL42[ril] as usize], LOW_MAX_NB, LOW_SHIFT);
        self.low.predict(dlow);

        // High band: SUBTRA, QUANTH, INVQAH, LOGSCH, SCALEH, then predict.
        let eh = saturate(xhigh - self.high.s);
        let wd = if eh >= 0 { eh } else { -(eh + 1) };
        let mih = if wd >= (564 * self.high.det) >> 12 {
            2
        } else {
            1
        };
        let ihigh = if eh < 0 { IHN[mih] } else { IHP[mih] };
        let dhigh = (self.high.det * QM2[ihigh as usize]) >> 15;
        self.high
            .scale(WH[RH2[ihigh as usize] as usize], HIGH_MAX_NB, HIGH_SHIFT);
        self.high.predict(dhigh);

        ((ihigh << 6) | ilow) as u8
    }
}

impl AudioEncoder for G722Encoder {
    /// Encodes pairs of 16 kHz samples. An odd sample at the end is held and
    /// paired with the first sample of the next call, so every input sample
    /// is consumed.
    fn encode(&mut self, pcm: &[i16], output: &mut Vec<u8>) -> usize {
        let mut samples = pcm.iter().copied();
        if let Some(first) = self.pending.take() {
            let Some(second) = samples.next() else {
                self.pending = Some(first);
                return 0;
            };
            output.push(self.encode_pair(first, second));
        }
        loop {
            match (samples.next(), samples.next()) {
                (Some(first), Some(second)) => output.push(self.encode_pair(first, second)),
                (Some(first), None) => {
                    self.pending = Some(first);
                    break;
                }
                _ => break,
            }
        }
        pcm.len()
    }

    fn payload_type(&self) -> u8 {
        CodecType::G722.payload_type()
    }

    fn codec_type(&self) -> CodecType {
        CodecType::G722
    }
}

/// G.722 decoder: one byte in, two 16 kHz samples out.
#[derive(Debug, Clone)]
pub struct G722Decoder {
    x: [i32; 24],
    low: Band,
    high: Band,
}

impl Default for G722Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl G722Decoder {
    /// A new decoder in its reset state.
    pub fn new() -> Self {
        Self {
            x: [0; 24],
            low: Band::new(32),
            high: Band::new(8),
        }
    }

    fn decode_byte(&mut self, code: u8, output: &mut Vec<i16>) {
        let code = i32::from(code);
        let ilow = code & 0x3F;
        let ihigh = (code >> 6) & 0x03;

        // Low band: INVQBL, RECONS, LIMIT; then adapt as the encoder did.
        let wd2 = (self.low.det * QM6[ilow as usize]) >> 15;
        let rlow = (self.low.s + wd2).clamp(-16384, 16383);
        let ril = (ilow >> 2) as usize;
        let dlow = (self.low.det * QM4[ril]) >> 15;
        self.low
            .scale(WL[RL42[ril] as usize], LOW_MAX_NB, LOW_SHIFT);
        self.low.predict(dlow);

        // High band: INVQAH, RECONS, LIMIT, adapt.
        let dhigh = (self.high.det * QM2[ihigh as usize]) >> 15;
        let rhigh = (dhigh + self.high.s).clamp(-16384, 16383);
        self.high
            .scale(WH[RH2[ihigh as usize] as usize], HIGH_MAX_NB, HIGH_SHIFT);
        self.high.predict(dhigh);

        // Receive QMF.
        self.x.copy_within(2.., 0);
        self.x[22] = rlow + rhigh;
        self.x[23] = rlow - rhigh;
        let (mut out1, mut out2) = (0, 0);
        for i in 0..12 {
            out2 += self.x[2 * i] * QMF[i];
            out1 += self.x[2 * i + 1] * QMF[11 - i];
        }
        output.push(saturate(out1 >> 11) as i16);
        output.push(saturate(out2 >> 11) as i16);
    }
}

impl AudioDecoder for G722Decoder {
    fn decode(&mut self, encoded: &[u8], output: &mut Vec<i16>) {
        output.reserve(encoded.len() * 2);
        for &code in encoded {
            self.decode_byte(code, output);
        }
    }

    fn codec_type(&self) -> CodecType {
        CodecType::G722
    }
}

//! Adaptive postfilter: long-term (pitch) postfilter, short-term formant
//! postfilter, tilt compensation and gain control (POSTFILT.C).

use super::basic_op::{
    MIN_32, add, div_s, extract_h, l_deposit_l, l_mac, l_mult, l_shl, l_shr, l_sub, mult, norm_l,
    round, shr, sub,
};
use super::dsp::{inv_sqrt, residu, syn_filt};
use super::lpc::weight_az;
use super::{L_FRAME, L_SUBFR, M, MP1, PIT_MAX};

const L_H: usize = 22;
const GAMMAP: i16 = 16384;
const INV_GAMMAP: i16 = 21845;
const GAMMAP_2: i16 = 10923;
const GAMMA2_PST: i16 = 18022;
const GAMMA1_PST: i16 = 22938;
const MU: i16 = 26214;
const AGC_FAC: i16 = 29491;
const AGC_FAC1: i16 = 32767 - AGC_FAC;

pub struct PostFilter {
    /// LP residual of the synthesis, `PIT_MAX` samples of history then the
    /// current subframe.
    res2: [i16; PIT_MAX + L_SUBFR],
    scal_res2: [i16; PIT_MAX + L_SUBFR],
    mem_syn_pst: [i16; M],
    /// The last `M` unfiltered synthesis samples of the previous frame.
    syn_hist: [i16; M],
    mem_pre: i16,
    past_gain: i16,
}

impl PostFilter {
    pub const fn new() -> Self {
        Self {
            res2: [0; PIT_MAX + L_SUBFR],
            scal_res2: [0; PIT_MAX + L_SUBFR],
            mem_syn_pst: [0; M],
            syn_hist: [0; M],
            mem_pre: 0,
            past_gain: 4096,
        }
    }

    /// Postfilter one frame of synthesis in place, given both subframes'
    /// LP filters and pitch lags.
    pub fn run(&mut self, syn: &mut [i16; L_FRAME], az_4: &[i16; 2 * MP1], t: [i16; 2]) {
        let mut buf = [0i16; M + L_FRAME];
        buf[..M].copy_from_slice(&self.syn_hist);
        buf[M..].copy_from_slice(syn);

        let mut syn_pst = [0i16; L_FRAME];
        let mut res2_pst = [0i16; L_SUBFR];
        let mut ap3 = [0i16; MP1];
        let mut ap4 = [0i16; MP1];

        for (sf, i_subfr) in (0..L_FRAME).step_by(L_SUBFR).enumerate() {
            let az = &az_4[sf * MP1..(sf + 1) * MP1];
            let mut t0_min = sub(t[sf], 3);
            let mut t0_max = add(t0_min, 6);
            if sub(t0_max, PIT_MAX as i16) > 0 {
                t0_max = PIT_MAX as i16;
                t0_min = sub(t0_max, 6);
            }

            weight_az(az, GAMMA2_PST, &mut ap3);
            weight_az(az, GAMMA1_PST, &mut ap4);

            residu(&ap3, &buf, M + i_subfr, &mut self.res2[PIT_MAX..]);
            for j in 0..L_SUBFR {
                self.scal_res2[PIT_MAX + j] = shr(self.res2[PIT_MAX + j], 2);
            }

            self.pit_pst_filt(t0_min, t0_max, &mut res2_pst);

            // Tilt of the formant postfilter's impulse response.
            let mut h_in = [0i16; L_H];
            h_in[..MP1].copy_from_slice(&ap3);
            let mut h = [0i16; L_H];
            let mut zero_mem = [0i16; M];
            syn_filt(&ap4, &h_in, &mut h, &mut zero_mem, false);

            let mut acc = l_mult(h[0], h[0]);
            for &v in &h[1..] {
                acc = l_mac(acc, v, v);
            }
            let temp1 = extract_h(acc);
            acc = l_mult(h[0], h[1]);
            for i in 1..L_H - 1 {
                acc = l_mac(acc, h[i], h[i + 1]);
            }
            let mut temp2 = extract_h(acc);
            temp2 = if temp2 <= 0 {
                0
            } else {
                div_s(mult(temp2, MU), temp1)
            };

            self.preemphasis(&mut res2_pst, temp2);

            syn_filt(
                &ap4,
                &res2_pst,
                &mut syn_pst[i_subfr..i_subfr + L_SUBFR],
                &mut self.mem_syn_pst,
                true,
            );

            self.agc(
                &buf[M + i_subfr..M + i_subfr + L_SUBFR],
                &mut syn_pst[i_subfr..i_subfr + L_SUBFR],
            );

            self.res2.copy_within(L_SUBFR.., 0);
            self.scal_res2.copy_within(L_SUBFR.., 0);
        }

        self.syn_hist.copy_from_slice(&buf[L_FRAME..]);
        *syn = syn_pst;
    }

    /// Harmonic postfilter on the current residual subframe.
    fn pit_pst_filt(&self, t0_min: i16, t0_max: i16, signal_pst: &mut [i16; L_SUBFR]) {
        const B: usize = PIT_MAX;
        let scal = &self.scal_res2;
        let signal = &self.res2;

        let mut cor_max = MIN_32;
        let mut t0 = t0_min;
        for i in t0_min..=t0_max {
            let lag = i as usize;
            let mut corr = 0;
            for j in 0..L_SUBFR {
                corr = l_mac(corr, scal[B + j], scal[B + j - lag]);
            }
            if l_sub(corr, cor_max) > 0 {
                cor_max = corr;
                t0 = i;
            }
        }
        let t0u = t0 as usize;

        let mut ener = 1;
        for j in 0..L_SUBFR {
            let v = scal[B + j - t0u];
            ener = l_mac(ener, v, v);
        }
        let mut ener0 = 1;
        for j in 0..L_SUBFR {
            let v = scal[B + j];
            ener0 = l_mac(ener0, v, v);
        }
        cor_max = cor_max.max(0);

        let temp = cor_max.max(ener).max(ener0);
        let j = norm_l(temp);
        let mut cmax = round(l_shl(cor_max, j));
        let mut en = round(l_shl(ener, j));
        let en0 = round(l_shl(ener0, j));

        let temp = l_sub(l_mult(cmax, cmax), l_shr(l_mult(en, en0), 1));
        if temp < 0 {
            signal_pst.copy_from_slice(&signal[B..]);
            return;
        }

        let (g0, gain) = if sub(cmax, en) > 0 {
            (INV_GAMMAP, GAMMAP_2)
        } else {
            cmax = shr(mult(cmax, GAMMAP), 1);
            en = shr(en, 1);
            let i = add(cmax, en);
            if i > 0 {
                let gain = div_s(cmax, i);
                (sub(32767, gain), gain)
            } else {
                (32767, 0)
            }
        };

        for i in 0..L_SUBFR {
            signal_pst[i] = add(mult(g0, signal[B + i]), mult(gain, signal[B + i - t0u]));
        }
    }

    fn preemphasis(&mut self, signal: &mut [i16; L_SUBFR], g: i16) {
        let temp = signal[L_SUBFR - 1];
        for i in (1..L_SUBFR).rev() {
            signal[i] = sub(signal[i], mult(g, signal[i - 1]));
        }
        signal[0] = sub(signal[0], mult(g, self.mem_pre));
        self.mem_pre = temp;
    }

    /// Scale `sig_out` to the energy of `sig_in`, smoothing the gain.
    fn agc(&mut self, sig_in: &[i16], sig_out: &mut [i16]) {
        let energy = |x: &[i16]| {
            let mut s = 0;
            for &v in x {
                let v = shr(v, 2);
                s = l_mac(s, v, v);
            }
            s
        };

        let s = energy(sig_out);
        if s == 0 {
            self.past_gain = 0;
            return;
        }
        let mut exp = sub(norm_l(s), 1);
        let gain_out = round(l_shl(s, exp));

        let s = energy(sig_in);
        let g0 = if s == 0 {
            0
        } else {
            let i = norm_l(s);
            let gain_in = round(l_shl(s, i));
            exp = sub(exp, i);
            let mut s = l_deposit_l(div_s(gain_out, gain_in));
            s = l_shl(s, 7);
            s = l_shr(s, exp);
            s = inv_sqrt(s);
            let i = round(l_shl(s, 9));
            mult(i, AGC_FAC1)
        };

        let mut gain = self.past_gain;
        for v in sig_out.iter_mut() {
            gain = mult(gain, AGC_FAC);
            gain = add(gain, g0);
            *v = extract_h(l_shl(l_mult(*v, gain), 3));
        }
        self.past_gain = gain;
    }
}

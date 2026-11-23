//! Annex B voice activity detector (VAD.C).
//!
//! Classifies each frame as speech or noise from its full- and low-band
//! energy, spectral distortion against the running noise LSFs, and
//! zero-crossing rate, each compared with a slowly adapting background
//! estimate.

use super::M;
use super::basic_op::{
    MAX_16, abs_s, add, extract_h, l_add, l_comp, l_deposit_h, l_mac, l_mult, l_shl, l_shr,
    mpy_32_16, mult, shr, sub,
};
use super::dsp::log2;
use super::lpc::NP;
use super::tables::{FACTOR_FX, LBF_CORR, SHIFT_FX};

const INIT_FRAME: i16 = 32;
const INIT_COUNT: i16 = 20;
const ZC_START: usize = 120;
const ZC_END: usize = 200;

/// Per-frame inputs the VAD reads from the LP analysis.
pub struct VadInput<'a> {
    /// Second reflection coefficient.
    pub rc: i16,
    /// Normalised LSFs (Q15).
    pub lsf: &'a [i16; M],
    pub r_h: &'a [i16; NP + 1],
    pub r_l: &'a [i16; NP + 1],
    pub exp_r0: i16,
    /// The pre-processed speech window the LP analysis used.
    pub sigpp: &'a [i16],
    /// Frame counter, starting at 1 (wraps from 32767 to 256).
    pub frm_count: i16,
    pub prev_marker: bool,
    pub pprev_marker: bool,
}

pub struct Vad {
    mean_lsf: [i16; M],
    min_buffer: [i16; 16],
    prev_min: i16,
    next_min: i16,
    min: i16,
    mean_e: i16,
    mean_se: i16,
    mean_sle: i16,
    mean_szc: i16,
    prev_energy: i16,
    count_sil: i16,
    count_update: i16,
    count_ext: i16,
    flag: bool,
    v_flag: bool,
    less_count: i16,
}

impl Vad {
    pub const fn new() -> Self {
        Self {
            mean_lsf: [0; M],
            min_buffer: [0; 16],
            prev_min: 0,
            next_min: 0,
            min: MAX_16,
            mean_e: 0,
            mean_se: 0,
            mean_sle: 0,
            mean_szc: 0,
            prev_energy: 0,
            count_sil: 0,
            count_update: 0,
            count_ext: 0,
            flag: true,
            v_flag: false,
            less_count: 0,
        }
    }

    /// `true` for active speech.
    pub fn detect(&mut self, v: &VadInput) -> bool {
        let frm_count = v.frm_count;

        // Full-band energy.
        let mut acc0 = l_comp(v.r_h[0], v.r_l[0]);
        let energy = log_energy(acc0, v.exp_r0);

        // Low-band energy.
        acc0 = 0;
        for i in 1..=NP {
            acc0 = l_mac(acc0, v.r_h[i], LBF_CORR[i]);
        }
        acc0 = l_shl(acc0, 1);
        acc0 = l_mac(acc0, v.r_h[0], LBF_CORR[0]);
        let energy_low = log_energy(acc0, v.exp_r0);

        // Spectral distortion.
        acc0 = 0;
        for i in 0..M {
            let j = sub(v.lsf[i], self.mean_lsf[i]);
            acc0 = l_mac(acc0, j, j);
        }
        let sd = extract_h(acc0);

        // Zero crossings.
        let mut zc: i16 = 0;
        for i in ZC_START + 1..=ZC_END {
            if mult(v.sigpp[i - 1], v.sigpp[i]) < 0 {
                zc = add(zc, 410);
            }
        }

        // Track the minimum energy over the last 128 frames.
        if sub(frm_count, 129) < 0 {
            if sub(energy, self.min) < 0 {
                self.min = energy;
                self.prev_min = energy;
            }
            if frm_count.trailing_zeros() >= 3 {
                let i = sub(shr(frm_count, 3), 1) as usize;
                self.min_buffer[i] = self.min;
                self.min = MAX_16;
            }
        }
        if frm_count.trailing_zeros() >= 3 {
            self.prev_min = self.buffer_min();
        }
        if sub(frm_count, 129) >= 0 {
            if (frm_count & 0x0007) ^ 0x0001 == 0 {
                self.min = self.prev_min;
                self.next_min = MAX_16;
            }
            if sub(energy, self.min) < 0 {
                self.min = energy;
            }
            if sub(energy, self.next_min) < 0 {
                self.next_min = energy;
            }
            if frm_count.trailing_zeros() >= 3 {
                self.min_buffer.copy_within(1.., 0);
                self.min_buffer[15] = self.next_min;
                self.prev_min = self.buffer_min();
            }
        }

        let mut marker = false;
        if sub(frm_count, INIT_FRAME) <= 0 {
            if sub(energy, 3072) < 0 {
                marker = false;
                self.less_count += 1;
            } else {
                marker = true;
                self.mean_e = extract_h(l_mac(l_deposit_h(self.mean_e), energy, 1024));
                self.mean_szc = extract_h(l_mac(l_deposit_h(self.mean_szc), zc, 1024));
                for i in 0..M {
                    self.mean_lsf[i] =
                        extract_h(l_mac(l_deposit_h(self.mean_lsf[i]), v.lsf[i], 1024));
                }
            }
        }

        if sub(frm_count, INIT_FRAME) >= 0 {
            if frm_count == INIT_FRAME {
                let lc = self.less_count as usize;
                let scale = |x: i16| extract_h(l_shl(l_mult(x, FACTOR_FX[lc]), SHIFT_FX[lc]));
                self.mean_e = scale(self.mean_e);
                self.mean_szc = scale(self.mean_szc);
                for v in &mut self.mean_lsf {
                    *v = scale(*v);
                }
                self.mean_se = sub(self.mean_e, 2048);
                self.mean_sle = sub(self.mean_e, 2458);
            }

            let d_se = sub(self.mean_se, energy);
            let d_sle = sub(self.mean_sle, energy_low);
            let d_szc = sub(self.mean_szc, zc);

            marker = if sub(energy, 3072) < 0 {
                false
            } else {
                make_dec(d_sle, d_se, sd, d_szc)
            };

            self.v_flag = false;
            if v.prev_marker && !marker && add(d_se, 410) < 0 && sub(energy, 3072) > 0 {
                marker = true;
                self.v_flag = true;
            }

            if self.flag {
                if v.pprev_marker
                    && v.prev_marker
                    && !marker
                    && sub(abs_s(sub(self.prev_energy, energy)), 614) <= 0
                {
                    self.count_ext += 1;
                    marker = true;
                    self.v_flag = true;
                    if sub(self.count_ext, 4) <= 0 {
                        self.flag = true;
                    } else {
                        self.count_ext = 0;
                        self.flag = false;
                    }
                }
            } else {
                self.flag = true;
            }

            if !marker {
                self.count_sil += 1;
            }
            if marker && sub(self.count_sil, 10) > 0 && sub(sub(energy, self.prev_energy), 614) <= 0
            {
                marker = false;
                self.count_sil = 0;
            }
            if marker {
                self.count_sil = 0;
            }

            if sub(sub(energy, 614), self.mean_se) < 0
                && sub(frm_count, 128) > 0
                && !self.v_flag
                && sub(v.rc, 19661) < 0
            {
                marker = false;
            }

            if sub(sub(energy, 614), self.mean_se) < 0 && sub(v.rc, 24576) < 0 && sub(sd, 83) < 0 {
                self.count_update += 1;
                let (coef, c_coef, coefzc, c_coefzc, coefsd, c_coefsd) = match self.count_update {
                    n if sub(n, INIT_COUNT) < 0 => (24576, 8192, 26214, 6554, 19661, 13017),
                    n if sub(n, INIT_COUNT + 10) < 0 => (31130, 1638, 30147, 2621, 21299, 11469),
                    n if sub(n, INIT_COUNT + 20) < 0 => (31785, 983, 30802, 1966, 22938, 9830),
                    n if sub(n, INIT_COUNT + 30) < 0 => (32440, 328, 31457, 1311, 24576, 8192),
                    n if sub(n, INIT_COUNT + 40) < 0 => (32604, 164, 32440, 328, 24576, 8192),
                    _ => (32604, 164, 32702, 66, 24576, 8192),
                };
                self.mean_se = extract_h(l_mac(l_mult(coef, self.mean_se), c_coef, energy));
                self.mean_sle = extract_h(l_mac(l_mult(coef, self.mean_sle), c_coef, energy_low));
                self.mean_szc = extract_h(l_mac(l_mult(coefzc, self.mean_szc), c_coefzc, zc));
                for i in 0..M {
                    self.mean_lsf[i] =
                        extract_h(l_mac(l_mult(coefsd, self.mean_lsf[i]), c_coefsd, v.lsf[i]));
                }
            }

            if sub(frm_count, 128) > 0
                && ((sub(self.mean_se, self.min) < 0 && sub(sd, 83) < 0)
                    || sub(self.mean_se, self.min) > 2048)
            {
                self.mean_se = self.min;
                self.count_update = 0;
            }
        }

        self.prev_energy = energy;
        marker
    }

    fn buffer_min(&self) -> i16 {
        let mut m = self.min_buffer[0];
        for &v in &self.min_buffer[1..] {
            if sub(v, m) < 0 {
                m = v;
            }
        }
        m
    }
}

/// `10 log10` of an autocorrelation energy, in the VAD's Q format.
fn log_energy(acc: i32, exp_r0: i16) -> i16 {
    let (exp, frac) = log2(acc);
    let mut acc0 = mpy_32_16(exp, frac, 9864);
    let i = sub(sub(exp_r0, 1), 1);
    acc0 = l_mac(acc0, 9864, i);
    acc0 = l_shl(acc0, 11);
    sub(extract_h(acc0), 4875)
}

/// The VAD's piecewise-linear decision boundaries; `true` for speech.
/// Each test is one boundary in the (SD, dSE, dSLE, dSZC) space.
fn make_dec(d_sle: i16, d_se: i16, sd: i16, d_szc: i16) -> bool {
    let sd_h = l_deposit_h(sd);
    let dse_h = l_deposit_h(d_se);
    let dsle_h = l_deposit_h(d_sle);
    // `(x * a + b * c) >> sh`, the shape most boundaries share.
    let line = |x: i16, a: i16, b: i16, c: i16, sh: i16| l_shr(l_mac(l_mult(x, a), b, c), sh);

    if l_add(line(d_szc, -14680, 8192, -28521, 8), sd_h) > 0 {
        return true;
    }
    if l_add(line(d_szc, 19065, 8192, -19446, 7), sd_h) > 0 {
        return true;
    }
    if l_add(line(d_szc, 20480, 8192, 16384, 2), dse_h) < 0 {
        return true;
    }
    if l_add(line(d_szc, -16384, 8192, 19660, 2), dse_h) < 0 {
        return true;
    }
    if l_mac(l_mult(d_se, 32767), 1024, 30802) < 0 {
        return true;
    }
    if l_mac(l_mac(l_mult(sd, -28160), 64, 19988), d_se, 512) < 0 {
        return true;
    }
    if l_mac(l_mult(sd, 32767), 32, -30199) > 0 {
        return true;
    }
    if l_add(line(d_szc, -20480, 8192, 22938, 2), dse_h) < 0 {
        return true;
    }
    if l_add(line(d_szc, 23831, 4096, 31576, 2), dse_h) < 0 {
        return true;
    }
    if l_mac(l_mult(d_se, 32767), 2048, 17367) < 0 {
        return true;
    }
    if l_mac(l_mac(l_mult(sd, -22400), 32, 25395), d_sle, 256) < 0 {
        return true;
    }
    if l_add(l_mac(l_mult(d_se, -30427), 256, -29959), dsle_h) > 0 {
        return true;
    }
    if l_add(l_mac(l_mult(d_se, -23406), 512, 28087), dsle_h) < 0 {
        return true;
    }
    l_mac(l_mac(l_mult(d_se, 24576), 1024, 29491), d_sle, 16384) < 0
}

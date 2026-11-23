//! Pitch: open-loop and closed-loop search, the fractional-delay
//! predictor, lag coding and the pitch parity bit (PITCH_A.C, PRED_LT3.C,
//! DEC_LAG3.C, P_PARITY.C), plus the encoder's taming test (TAMING.C).

use super::acelp::cor_h_x;
use super::basic_op::{
    MIN_32, abs_s, add, clear_overflow, div_s, extract_l, l_add, l_extract, l_mac, l_shl, l_sub,
    mpy_32, mpy_32_16, mult, negate, norm_l, overflow, round, shl, shr, sub,
};
use super::dsp::inv_sqrt;
use super::tables::{INTER_3L, TAB_ZONE};
use super::{L_FRAME, L_INTERPOL, L_SUBFR, PIT_MAX, PIT_MIN};

const UP_SAMP: i16 = 3;
const L_INTER10: usize = 10;

/// Open-loop pitch estimate over the weighted speech `buf`, where
/// `buf[PIT_MAX..]` is the current frame and `buf[..PIT_MAX]` its history.
pub fn pitch_ol_fast(buf: &[i16; L_FRAME + PIT_MAX]) -> i16 {
    const BASE: usize = PIT_MAX;
    let mut scaled = [0i16; L_FRAME + PIT_MAX];

    clear_overflow();
    let mut sum = 0;
    for i in (0..L_FRAME + PIT_MAX).step_by(2) {
        sum = l_mac(sum, buf[i], buf[i]);
    }
    if overflow() {
        for (s, &v) in scaled.iter_mut().zip(buf) {
            *s = shr(v, 3);
        }
    } else if l_sub(sum, 1_048_576) < 0 {
        for (s, &v) in scaled.iter_mut().zip(buf) {
            *s = shl(v, 3);
        }
    } else {
        scaled = *buf;
    }

    let corr = |lag: usize| {
        let mut sum = 0;
        for j in (0..L_FRAME).step_by(2) {
            sum = l_mac(sum, scaled[BASE + j], scaled[BASE + j - lag]);
        }
        sum
    };
    // Normalised correlation at `lag` for the best correlation `max`.
    let normalise = |max: i32, lag: usize| {
        let mut sum = 1;
        for j in (0..L_FRAME).step_by(2) {
            let v = scaled[BASE + j - lag];
            sum = l_mac(sum, v, v);
        }
        let sum = inv_sqrt(sum);
        let (max_h, max_l) = l_extract(max);
        let (ener_h, ener_l) = l_extract(sum);
        extract_l(mpy_32(max_h, max_l, ener_h, ener_l))
    };
    let search = |lags: &mut dyn Iterator<Item = usize>, first: usize| {
        let mut max = MIN_32;
        let mut t = first;
        for i in lags {
            let sum = corr(i);
            if l_sub(sum, max) > 0 {
                max = sum;
                t = i;
            }
        }
        (max, t)
    };

    let (max, t1) = search(&mut (20..40), 20);
    let mut max1 = normalise(max, t1);
    let mut t1 = t1 as i16;

    let (max, t2) = search(&mut (40..80), 40);
    let mut max2 = normalise(max, t2);
    let t2 = t2 as i16;

    let (mut max, mut t3) = search(&mut (80..143).step_by(2), 80);
    let i = t3;
    let sum = corr(i + 1);
    if l_sub(sum, max) > 0 {
        max = sum;
        t3 = i + 1;
    }
    let sum = corr(i - 1);
    if l_sub(sum, max) > 0 {
        max = sum;
        t3 = i - 1;
    }
    let max3 = normalise(max, t3);
    let t3 = t3 as i16;

    // Favour submultiples of the longer lags.
    let mut i = sub(shl(t2, 1), t3);
    if sub(abs_s(i), 5) < 0 {
        max2 = add(max2, shr(max3, 2));
    }
    i = add(i, t2);
    if sub(abs_s(i), 7) < 0 {
        max2 = add(max2, shr(max3, 2));
    }
    let mut i = sub(shl(t1, 1), t2);
    if sub(abs_s(i), 5) < 0 {
        max1 = add(max1, mult(max2, 6554));
    }
    i = add(i, t1);
    if sub(abs_s(i), 7) < 0 {
        max1 = add(max1, mult(max2, 6554));
    }

    if sub(max1, max2) < 0 {
        max1 = max2;
        t1 = t2;
    }
    if sub(max1, max3) < 0 {
        t1 = t3;
    }
    t1
}

fn dot_product(x: &[i16], y: &[i16]) -> i32 {
    x.iter().zip(y).fold(0, |s, (&a, &b)| l_mac(s, a, b))
}

/// Adaptive-codebook vector: `buf[pos..pos + l_subfr]` from the past
/// excitation at delay `t0` + `frac`/3, using the 1/3-resolution
/// interpolation filter. Reads samples it may itself have just written
/// (for lags shorter than the subframe), as the reference does.
pub fn pred_lt_3(buf: &mut [i16], pos: usize, t0: i16, frac: i16, l_subfr: usize) {
    let mut x0 = pos - t0 as usize;
    let mut frac = negate(frac);
    if frac < 0 {
        frac = add(frac, UP_SAMP);
        x0 -= 1;
    }
    let c1 = frac as usize;
    let c2 = sub(UP_SAMP, frac) as usize;
    for j in 0..l_subfr {
        let x1 = x0;
        x0 += 1;
        let x2 = x0;
        let mut s = 0;
        for i in 0..L_INTER10 {
            let k = i * UP_SAMP as usize;
            s = l_mac(s, buf[x1 - i], INTER_3L[c1 + k]);
            s = l_mac(s, buf[x2 + i], INTER_3L[c2 + k]);
        }
        buf[pos + j] = round(s);
    }
}

/// Closed-loop fractional pitch search around the open-loop estimate.
/// `exc[pos..]` is the subframe's excitation, overwritten with the chosen
/// adaptive-codebook vector. Returns the integer lag and sets `pit_frac`.
pub fn pitch_fr3_fast(
    exc: &mut [i16],
    pos: usize,
    xn: &[i16],
    h: &[i16],
    t0_min: i16,
    t0_max: i16,
    i_subfr: usize,
    pit_frac: &mut i16,
) -> i16 {
    let mut dn = [0i16; L_SUBFR];
    cor_h_x(h, xn, &mut dn);

    let mut max = MIN_32;
    let mut t0 = t0_min;
    for t in t0_min..=t0_max {
        let start = pos - t as usize;
        let corr = dot_product(&dn, &exc[start..start + L_SUBFR]);
        if l_sub(corr, max) > 0 {
            max = corr;
            t0 = t;
        }
    }

    pred_lt_3(exc, pos, t0, 0, L_SUBFR);
    max = dot_product(&dn, &exc[pos..pos + L_SUBFR]);
    *pit_frac = 0;

    if i_subfr == 0 && sub(t0, 84) > 0 {
        return t0;
    }

    let mut exc_tmp = [0i16; L_SUBFR];
    exc_tmp.copy_from_slice(&exc[pos..pos + L_SUBFR]);

    pred_lt_3(exc, pos, t0, -1, L_SUBFR);
    let corr = dot_product(&dn, &exc[pos..pos + L_SUBFR]);
    if l_sub(corr, max) > 0 {
        max = corr;
        *pit_frac = -1;
        exc_tmp.copy_from_slice(&exc[pos..pos + L_SUBFR]);
    }

    pred_lt_3(exc, pos, t0, 1, L_SUBFR);
    let corr = dot_product(&dn, &exc[pos..pos + L_SUBFR]);
    if l_sub(corr, max) > 0 {
        *pit_frac = 1;
    } else {
        exc[pos..pos + L_SUBFR].copy_from_slice(&exc_tmp);
    }
    t0
}

/// Adaptive-codebook gain (Q14, at most 1.2), and the correlations the
/// gain quantiser needs.
pub fn g_pitch(xn: &[i16], y1: &[i16], g_coeff: &mut [i16; 4]) -> i16 {
    let mut scaled_y1 = [0i16; L_SUBFR];
    for (s, &v) in scaled_y1.iter_mut().zip(y1) {
        *s = shr(v, 2);
    }

    clear_overflow();
    let mut s = 1;
    for &v in &y1[..L_SUBFR] {
        s = l_mac(s, v, v);
    }
    let (yy, exp_yy) = if overflow() {
        let mut s = 1;
        for &v in &scaled_y1 {
            s = l_mac(s, v, v);
        }
        let e = norm_l(s);
        (round(l_shl(s, e)), sub(e, 4))
    } else {
        let e = norm_l(s);
        (round(l_shl(s, e)), e)
    };

    clear_overflow();
    let mut s = 0;
    for i in 0..L_SUBFR {
        s = l_mac(s, xn[i], y1[i]);
    }
    let (xy, exp_xy) = if overflow() {
        let mut s = 0;
        for i in 0..L_SUBFR {
            s = l_mac(s, xn[i], scaled_y1[i]);
        }
        let e = norm_l(s);
        (round(l_shl(s, e)), sub(e, 2))
    } else {
        let e = norm_l(s);
        (round(l_shl(s, e)), e)
    };

    g_coeff[0] = yy;
    g_coeff[1] = sub(15, exp_yy);
    g_coeff[2] = xy;
    g_coeff[3] = sub(15, exp_xy);

    if xy <= 0 {
        g_coeff[3] = -15;
        return 0;
    }

    let mut gain = div_s(shr(xy, 1), yy);
    gain = shr(gain, sub(exp_xy, exp_yy));
    gain.min(19661)
}

/// Encode the pitch lag; for the first subframe also narrows the search
/// range used by the second.
pub fn enc_lag3(t0: i16, t0_frac: i16, t0_min: &mut i16, t0_max: &mut i16, first: bool) -> i16 {
    if first {
        let index = if sub(t0, 85) <= 0 {
            let i = add(add(t0, t0), t0);
            add(sub(i, 58), t0_frac)
        } else {
            add(t0, 112)
        };
        *t0_min = sub(t0, 5).max(PIT_MIN as i16);
        *t0_max = add(*t0_min, 9);
        if sub(*t0_max, PIT_MAX as i16) > 0 {
            *t0_max = PIT_MAX as i16;
            *t0_min = sub(*t0_max, 9);
        }
        index
    } else {
        let mut i = sub(t0, *t0_min);
        i = add(add(i, i), i);
        add(add(i, 2), t0_frac)
    }
}

/// Decode a pitch index. For the second subframe `t0` holds the first
/// subframe's lag on entry.
pub fn dec_lag3(index: i16, first: bool, t0: &mut i16, t0_frac: &mut i16) {
    if first {
        if sub(index, 197) < 0 {
            *t0 = add(mult(add(index, 2), 10923), 19);
            let i = add(add(*t0, *t0), *t0);
            *t0_frac = add(sub(index, i), 58);
        } else {
            *t0 = sub(index, 112);
            *t0_frac = 0;
        }
    } else {
        let mut t0_min = sub(*t0, 5).max(PIT_MIN as i16);
        if sub(add(t0_min, 9), PIT_MAX as i16) > 0 {
            t0_min = sub(PIT_MAX as i16, 9);
        }
        let i = sub(mult(add(index, 2), 10923), 1);
        *t0 = add(i, t0_min);
        let i = add(add(i, i), i);
        *t0_frac = sub(sub(index, 2), i);
    }
}

fn parity_sum(pitch_index: i16) -> i16 {
    let mut temp = shr(pitch_index, 1);
    let mut sum = 1;
    for _ in 0..=5 {
        temp = shr(temp, 1);
        sum = add(sum, temp & 1);
    }
    sum
}

/// Parity bit over the six most significant bits of the first pitch index.
pub fn parity_pitch(pitch_index: i16) -> i16 {
    parity_sum(pitch_index) & 1
}

/// 1 when the received parity bit does not match.
pub fn check_parity_pitch(pitch_index: i16, parity: i16) -> i16 {
    add(parity_sum(pitch_index), parity) & 1
}

const L_THRESH_ERR: i32 = 983_040_000;

/// Excitation error tracker that decides when to "tame" the pitch gain to
/// stop the encoder's synthesis filter from going unstable.
pub struct Taming {
    l_exc_err: [i32; 4],
}

impl Taming {
    pub const fn new() -> Self {
        Self {
            l_exc_err: [0x4000; 4],
        }
    }

    /// Whether the pitch gain must be limited for lag `t0` + `t0_frac`.
    pub fn test_err(&self, t0: i16, t0_frac: i16) -> bool {
        let t1 = if t0_frac > 0 { add(t0, 1) } else { t0 };
        let i = sub(t1, (L_SUBFR + L_INTER10) as i16).max(0);
        let zone1 = TAB_ZONE[i as usize];
        let i = add(t1, (L_INTER10 - 2) as i16);
        let zone2 = TAB_ZONE[i as usize];

        let mut l_maxloc = -1;
        for i in (zone1..=zone2).rev() {
            let v = self.l_exc_err[i as usize];
            if l_sub(v, l_maxloc) > 0 {
                l_maxloc = v;
            }
        }
        l_sub(l_maxloc, L_THRESH_ERR) > 0
    }

    pub fn update(&mut self, gain_pit: i16, t0: i16) {
        let step = |prev: i32| {
            let (hi, lo) = l_extract(prev);
            let t = l_shl(mpy_32_16(hi, lo, gain_pit), 1);
            l_add(0x4000, t)
        };
        let mut l_worst = -1;
        let n = sub(t0, L_SUBFR as i16);
        if n < 0 {
            let t = step(self.l_exc_err[0]);
            if l_sub(t, l_worst) > 0 {
                l_worst = t;
            }
            let t = step(t);
            if l_sub(t, l_worst) > 0 {
                l_worst = t;
            }
        } else {
            let zone1 = TAB_ZONE[n as usize];
            let zone2 = TAB_ZONE[sub(t0, 1) as usize];
            for i in zone1..=zone2 {
                let t = step(self.l_exc_err[i as usize]);
                if l_sub(t, l_worst) > 0 {
                    l_worst = t;
                }
            }
        }
        self.l_exc_err.copy_within(0..3, 1);
        self.l_exc_err[0] = l_worst;
    }
}

/// Length of the excitation history the predictor needs.
pub const EXC_HISTORY: usize = PIT_MAX + L_INTERPOL;

//! Gain prediction, the conjugate-structure gain VQ and its decoding
//! (GAINPRED.C, QUA_GAIN.C, DEC_GAIN.C, and `Corr_xy2` from COR_FUNC.C).

use super::L_SUBFR;
use super::basic_op::{
    MAX_32, add, div_s, extract_h, extract_l, l_add, l_comp, l_deposit_h, l_deposit_l, l_extract,
    l_mac, l_mult, l_shl, l_shr, l_sub, mpy_32_16, mult, negate, norm_l, round, shl, shr, sub,
};
use super::dsp::{log2, pow2};
use super::tables::{COEF, GBK1, GBK2, IMAP1, IMAP2, L_COEF, MAP1, MAP2, PRED, THR1, THR2};

const NCODE1: usize = 8;
const NCODE2: usize = 16;
const NCODE2_B: i16 = 4;
const NCAN1: usize = 4;
const NCAN2: usize = 8;
const INV_COEF: i16 = -17103;
const GPCLIP2: i16 = 481;
const GP0999: i16 = 16383;
/// Q-format alignment of the gain thresholds in `gbk_presel`.
const SFT_Y: i16 = (14 + 4 + 1) - 16;
const SFT_X: i16 = (15 + 4 + 1) - 15;

/// MA-predicted codebook gain from the past quantised energies, as
/// (mantissa, exponent).
fn gain_predict(past_qua_en: [i16; 4], code: &[i16]) -> (i16, i16) {
    let mut t = 0;
    for &c in &code[..L_SUBFR] {
        t = l_mac(t, c, c);
    }
    let (exp, frac) = log2(t);
    t = mpy_32_16(exp, frac, -24660);
    t = l_mac(t, 32588, 32);
    t = l_shl(t, 10);
    for (&p, e) in PRED.iter().zip(past_qua_en) {
        t = l_mac(t, p, e);
    }
    let gcode0 = extract_h(t);
    t = l_mult(gcode0, 5439);
    t = l_shr(t, 8);
    let (exp, frac) = l_extract(t);
    (extract_l(pow2(14, frac)), sub(14, exp))
}

fn gain_update(past_qua_en: &mut [i16; 4], l_gbk12: i32) {
    past_qua_en.copy_within(0..3, 1);
    let (exp, frac) = log2(l_gbk12);
    let acc = l_comp(sub(exp, 13), frac);
    let tmp = extract_h(l_shl(acc, 13));
    past_qua_en[0] = mult(tmp, 24660);
}

fn gain_update_erasure(past_qua_en: &mut [i16; 4]) {
    let mut t = 0;
    for &e in past_qua_en.iter() {
        t = l_add(t, l_deposit_l(e));
    }
    let av = sub(extract_l(l_shr(t, 2)), 4096).max(-14336);
    past_qua_en.copy_within(0..3, 1);
    past_qua_en[0] = av;
}

/// Codebook gain from the table entries and the prediction.
fn code_gain(l_gbk12: i32, gcode0: i16, exp_gcode0: i16) -> i16 {
    let tmp = extract_l(l_shr(l_gbk12, 1));
    let acc = l_mult(tmp, gcode0);
    extract_h(l_shl(acc, add(negate(exp_gcode0), 4)))
}

fn gbk12(i1: usize, i2: usize) -> i32 {
    l_add(l_deposit_l(GBK1[i1][1]), l_deposit_l(GBK2[i2][1]))
}

/// The `<y2,y2>`, `-2<xn,y2>` and `2<y1,y2>` terms of the gain
/// quantiser's error, with their Q formats, into `g_coeff[2..5]`.
pub fn corr_xy2(xn: &[i16], y1: &[i16], y2: &[i16], g_coeff: &mut [i16; 5], exp_g: &mut [i16; 5]) {
    let mut scaled_y2 = [0i16; L_SUBFR];
    for (s, &v) in scaled_y2.iter_mut().zip(y2) {
        *s = shr(v, 3);
    }
    let corr = |a: &[i16]| {
        let mut acc = 1;
        for (&x, &y) in a.iter().zip(&scaled_y2) {
            acc = l_mac(acc, x, y);
        }
        let e = norm_l(acc);
        (round(l_shl(acc, e)), e)
    };

    let (y2y2, e) = corr(&scaled_y2);
    g_coeff[2] = y2y2;
    exp_g[2] = add(e, 19 - 16);

    let (xny2, e) = corr(xn);
    g_coeff[3] = negate(xny2);
    exp_g[3] = sub(add(e, 10 - 16), 1);

    let (y1y2, e) = corr(y1);
    g_coeff[4] = y1y2;
    exp_g[4] = sub(add(e, 10 - 16), 1);
}

/// Encoder side of the gain VQ.
pub struct GainQuantizer {
    past_qua_en: [i16; 4],
}

impl GainQuantizer {
    pub const fn new() -> Self {
        Self {
            past_qua_en: [-14336; 4],
        }
    }

    /// Quantise the pitch and code gains jointly. Returns the 7-bit index
    /// and sets `gain_pit` (Q14) and `gain_cod` (Q1).
    pub fn quantize(
        &mut self,
        code: &[i16],
        g_coeff: &[i16; 5],
        exp_coeff: &[i16; 5],
        gain_pit: &mut i16,
        gain_cod: &mut i16,
        tameflag: bool,
    ) -> i16 {
        let (gcode0, exp_gcode0) = gain_predict(self.past_qua_en, code);

        // Unquantised optimum gains, from the 2x2 normal equations.
        let mut l_tmp1 = l_mult(g_coeff[0], g_coeff[2]);
        let mut exp1 = add(add(exp_coeff[0], exp_coeff[2]), 1 - 2);
        let mut l_tmp2 = l_mult(g_coeff[4], g_coeff[4]);
        let mut exp2 = add(add(exp_coeff[4], exp_coeff[4]), 1);
        let (mut l_tmp, mut exp) = if sub(exp1, exp2) > 0 {
            (l_sub(l_shr(l_tmp1, sub(exp1, exp2)), l_tmp2), exp2)
        } else {
            (l_sub(l_tmp1, l_shr(l_tmp2, sub(exp2, exp1))), exp1)
        };
        let mut sft = norm_l(l_tmp);
        let denom = extract_h(l_shl(l_tmp, sft));
        let exp_denom = sub(add(exp, sft), 16);
        let inv_denom = negate(div_s(16384, denom));
        let exp_inv_denom = sub(14 + 15, exp_denom);

        let mut best_gain = [0i16; 2];

        l_tmp1 = l_mult(g_coeff[2], g_coeff[1]);
        exp1 = add(exp_coeff[2], exp_coeff[1]);
        l_tmp2 = l_mult(g_coeff[3], g_coeff[4]);
        exp2 = add(add(exp_coeff[3], exp_coeff[4]), 1);
        (l_tmp, exp) = cross_term(l_tmp1, exp1, l_tmp2, exp2);
        sft = norm_l(l_tmp);
        let nume = extract_h(l_shl(l_tmp, sft));
        let exp_nume = sub(add(exp, sft), 16);
        sft = sub(add(exp_nume, exp_inv_denom), 9 + 16 - 1);
        best_gain[0] = extract_h(l_shr(l_mult(nume, inv_denom), sft));
        if tameflag {
            best_gain[0] = best_gain[0].min(GPCLIP2);
        }

        l_tmp1 = l_mult(g_coeff[0], g_coeff[3]);
        exp1 = add(exp_coeff[0], exp_coeff[3]);
        l_tmp2 = l_mult(g_coeff[1], g_coeff[4]);
        exp2 = add(add(exp_coeff[1], exp_coeff[4]), 1);
        (l_tmp, exp) = cross_term(l_tmp1, exp1, l_tmp2, exp2);
        sft = norm_l(l_tmp);
        let nume = extract_h(l_shl(l_tmp, sft));
        let exp_nume = sub(add(exp, sft), 16);
        sft = sub(add(exp_nume, exp_inv_denom), 2 + 16 - 1);
        best_gain[1] = extract_h(l_shr(l_mult(nume, inv_denom), sft));

        let gcode0_org = if sub(exp_gcode0, 4) >= 0 {
            shr(gcode0, sub(exp_gcode0, 4))
        } else {
            extract_h(l_shl(l_deposit_l(gcode0), sub(4 + 16, exp_gcode0)))
        };

        let (cand1, cand2) = gbk_presel(best_gain, gcode0_org);

        // Bring the error terms to a common exponent.
        let exp_min = [
            add(exp_coeff[0], 13),
            add(exp_coeff[1], 14),
            add(exp_coeff[2], sub(shl(exp_gcode0, 1), 21)),
            add(exp_coeff[3], sub(exp_gcode0, 3)),
            add(exp_coeff[4], sub(exp_gcode0, 4)),
        ];
        let e_min = *exp_min.iter().min().expect("five exponents");
        let mut coeff = [0i16; 5];
        let mut coeff_lsf = [0i16; 5];
        for i in 0..5 {
            let j = sub(exp_min[i], e_min);
            let t = l_shr(l_deposit_h(g_coeff[i]), j);
            (coeff[i], coeff_lsf[i]) = l_extract(t);
        }

        let mut l_dist_min = MAX_32;
        let mut index1 = cand1;
        let mut index2 = cand2;
        for i in 0..NCAN1 {
            for j in 0..NCAN2 {
                let g_pitch = add(GBK1[cand1 + i][0], GBK2[cand2 + j][0]);
                if tameflag && g_pitch >= GP0999 {
                    continue;
                }
                let tmp = extract_l(l_shr(gbk12(cand1 + i, cand2 + j), 1));
                let g_code = mult(gcode0, tmp);
                let g2_pitch = mult(g_pitch, g_pitch);
                let g2_code = mult(g_code, g_code);
                let g_pit_cod = mult(g_code, g_pitch);
                let mut t = mpy_32_16(coeff[0], coeff_lsf[0], g2_pitch);
                t = l_add(t, mpy_32_16(coeff[1], coeff_lsf[1], g_pitch));
                t = l_add(t, mpy_32_16(coeff[2], coeff_lsf[2], g2_code));
                t = l_add(t, mpy_32_16(coeff[3], coeff_lsf[3], g_code));
                t = l_add(t, mpy_32_16(coeff[4], coeff_lsf[4], g_pit_cod));
                if l_sub(t, l_dist_min) < 0 {
                    l_dist_min = t;
                    index1 = cand1 + i;
                    index2 = cand2 + j;
                }
            }
        }

        *gain_pit = add(GBK1[index1][0], GBK2[index2][0]);
        let l_gbk12 = gbk12(index1, index2);
        *gain_cod = code_gain(l_gbk12, gcode0, exp_gcode0);
        gain_update(&mut self.past_qua_en, l_gbk12);
        add(MAP1[index1] * NCODE2 as i16, MAP2[index2])
    }
}

/// `tmp1/2 - tmp2/2` at the smaller of the two exponents, minus one.
fn cross_term(l_tmp1: i32, exp1: i16, l_tmp2: i32, exp2: i16) -> (i32, i16) {
    if sub(exp1, exp2) > 0 {
        (
            l_sub(l_shr(l_tmp1, add(sub(exp1, exp2), 1)), l_shr(l_tmp2, 1)),
            sub(exp2, 1),
        )
    } else {
        (
            l_sub(l_shr(l_tmp1, 1), l_shr(l_tmp2, add(sub(exp2, exp1), 1))),
            sub(exp1, 1),
        )
    }
}

/// Pick the first candidate of each gain codebook to search from.
fn gbk_presel(best_gain: [i16; 2], gcode0: i16) -> (usize, usize) {
    let l_cfbg = l_mult(COEF[0][0], best_gain[0]);
    let mut acc = l_shr(L_COEF[1][1], 15);
    acc = l_add(l_cfbg, acc);
    let mut acc_h = extract_h(acc);
    let mut l_preg = l_mult(acc_h, gcode0);
    acc = l_shl(l_deposit_l(best_gain[1]), 7);
    acc = l_sub(acc, l_preg);
    acc_h = extract_h(l_shl(acc, 2));
    let l_tmp_x = l_mult(acc_h, INV_COEF);

    acc = l_shr(L_COEF[0][1], 10);
    acc = l_sub(l_cfbg, acc);
    acc_h = extract_h(acc);
    acc_h = mult(acc_h, gcode0);
    let l_tmp = l_mult(acc_h, COEF[1][0]);
    l_preg = l_mult(COEF[0][0], best_gain[1]);
    acc = l_sub(l_tmp, l_shr(l_preg, 3));
    acc_h = extract_h(l_shl(acc, 2));
    let l_tmp_y = l_mult(acc_h, INV_COEF);

    // Advance past every threshold the gain lies beyond; which side counts
    // as "beyond" flips with the sign of the predicted gain.
    let beyond = |l: i32| if gcode0 > 0 { l > 0 } else { l < 0 };
    let mut cand1 = 0;
    while beyond(l_sub(l_tmp_y, l_shr(l_mult(THR1[cand1], gcode0), SFT_Y))) {
        cand1 += 1;
        if cand1 >= NCODE1 - NCAN1 {
            break;
        }
    }
    let mut cand2 = 0;
    while beyond(l_sub(l_tmp_x, l_shr(l_mult(THR2[cand2], gcode0), SFT_X))) {
        cand2 += 1;
        if cand2 >= NCODE2 - NCAN2 {
            break;
        }
    }
    (cand1, cand2)
}

/// Decoder side of the gain VQ.
pub struct GainDecoder {
    past_qua_en: [i16; 4],
}

impl GainDecoder {
    pub const fn new() -> Self {
        Self {
            past_qua_en: [-14336; 4],
        }
    }

    /// Decode the gains. On a bad frame the previous gains (passed in
    /// `gain_pit`/`gain_cod`) are attenuated instead.
    pub fn decode(
        &mut self,
        index: i16,
        code: &[i16],
        bfi: bool,
        gain_pit: &mut i16,
        gain_cod: &mut i16,
    ) {
        if bfi {
            *gain_pit = mult(*gain_pit, 29491).min(29491);
            *gain_cod = mult(*gain_cod, 32111);
            gain_update_erasure(&mut self.past_qua_en);
            return;
        }
        let index1 = IMAP1[shr(index, NCODE2_B) as usize] as usize;
        let index2 = IMAP2[(index & (NCODE2 as i16 - 1)) as usize] as usize;
        *gain_pit = add(GBK1[index1][0], GBK2[index2][0]);
        let (gcode0, exp_gcode0) = gain_predict(self.past_qua_en, code);
        let l_gbk12 = gbk12(index1, index2);
        *gain_cod = code_gain(l_gbk12, gcode0, exp_gcode0);
        gain_update(&mut self.past_qua_en, l_gbk12);
    }
}

//! Annex B discontinuous transmission and comfort noise: the encoder's
//! SID decision and quantisation, the decoder's SID decoding, and the
//! comfort-noise excitation both sides synthesise (DTX.C, DEC_SID.C,
//! QSIDGAIN.C, QSIDLSF.C, CALCEXC.C).

use super::basic_op::{
    MAX_16, abs_s, add, clear_overflow, extract_h, extract_l, l_add, l_deposit_l, l_extract, l_mac,
    l_mult, l_shl, l_shr, l_sub, mpy_32_16, mult, mult_r, negate, norm_l, norm_s, overflow, round,
    shl, shr, shr_r, sub,
};
use super::dsp::{Random, inv_sqrt, log2};
use super::lpc::{Levinson, az_lsp, int_qlpc, lsf_lsp2, lsp_lsf2};
use super::lsp::{
    FreqPrev, GAP3, L_LIMIT, M_LIMIT, MA_NP, MaPredictor, get_wegt, lsp_expand, lsp_prev_compose,
    lsp_prev_extract, lsp_prev_update, lsp_stability,
};
use super::pitch::{Taming, pred_lt_3};
use super::tables::{
    FACT, FG, LSPCB1, LSPCB2, MARG, MP, NOISE_FG_SUM, NOISE_FG_SUM_INV, PTRTAB_1, PTRTAB_2,
    TAB_SIDGAIN,
};
use super::{L_FRAME, L_SUBFR, M, MP1};

/// Seed of the comfort-noise generator, reset on every speech frame.
pub const INIT_SEED: i16 = 11111;

const FR_SID_MIN: i16 = 3;
const NB_SUMACF: usize = 3;
const NB_CURACF: usize = 2;
const NB_GAIN: usize = 2;
const FRAC_THRESH1: i16 = 4855;
const FRAC_THRESH2: i16 = 3161;
const A_GAIN0: i16 = 28672;
const A_GAIN1: i16 = 4096;
const FRAC1: i16 = 19043;
const K0: i16 = 24576;
const G_MAX: i16 = 5000;
const SIZ_SUMACF: usize = NB_SUMACF * MP1;
const SIZ_ACF: usize = NB_CURACF * MP1;

/// The two MA predictors used for noise: the first speech predictor, and
/// a blend of both.
fn noise_fg() -> [MaPredictor; 2] {
    let mut fg = [FG[0], [[0; M]; MA_NP]];
    for i in 0..MA_NP {
        for j in 0..M {
            let acc = l_mac(l_mult(FG[0][i][j], 19660), FG[1][i][j], 13107);
            fg[1][i][j] = extract_h(acc);
        }
    }
    fg
}

// ---------------------------------------------------------------------
// SID gain

/// Quantise the mean of `nb_ener` frame energies (or, with `nb_ener == 0`,
/// the one energy given) to a 5-bit SID gain index. Returns
/// `(quantised energy, index)`.
fn qua_sidgain(ener: &[i16], sh_ener: &[i16], nb_ener: usize) -> (i16, i16) {
    let (l_x, sh1) = if nb_ener == 0 {
        let acc = l_shl(l_deposit_l(ener[0]), sh_ener[0]);
        let (hi, lo) = l_extract(acc);
        (mpy_32_16(hi, lo, FACT[0]), 0)
    } else {
        let mut sh1 = sh_ener[0];
        for &s in &sh_ener[1..nb_ener] {
            if s < sh1 {
                sh1 = s;
            }
        }
        sh1 = add(sh1, 16 - MARG[nb_ener]);
        let mut l_x = 0;
        for i in 0..nb_ener {
            let temp = sub(sh1, sh_ener[i]);
            l_x = l_add(l_x, l_shl(l_deposit_l(ener[i]), temp));
        }
        let (hi, lo) = l_extract(l_x);
        (mpy_32_16(hi, lo, FACT[nb_ener]), sh1)
    };
    quant_energy(l_x, sh1)
}

fn quant_energy(l_x: i32, sh: i16) -> (i16, i16) {
    let (exp, frac) = log2(l_x);
    let temp = sub(exp, sh);
    let mut e_tmp = shl(temp, 10);
    e_tmp = add(e_tmp, mult_r(frac, 1024));

    if sub(e_tmp, -2721) <= 0 {
        return (-12, 0);
    }
    if sub(e_tmp, 22111) > 0 {
        return (66, 31);
    }
    if sub(e_tmp, 4762) <= 0 {
        e_tmp = add(e_tmp, 3401);
        let index = mult(e_tmp, 24).max(1);
        return (sub(shl(index, 2), 8), index);
    }
    e_tmp = sub(e_tmp, 340);
    let index = sub(shr(mult(e_tmp, 193), 2), 1).max(6);
    (add(shl(index, 1), 4), index)
}

// ---------------------------------------------------------------------
// SID LSFs

/// Quantise the noise LSPs for a SID frame. `ana` takes the predictor
/// index and the two codebook indices.
fn lsfq_noise(
    lsp: &[i16; M],
    lspq: &mut [i16; M],
    freq_prev: &mut FreqPrev,
    noise_fg: &[MaPredictor; 2],
    ana: &mut [i16],
) {
    let mut lsf = [0i16; M];
    lsp_lsf2(lsp, &mut lsf);
    if lsf[0] < L_LIMIT {
        lsf[0] = L_LIMIT;
    }
    for i in 0..M - 1 {
        if sub(lsf[i + 1], lsf[i]) < 2 * GAP3 {
            lsf[i + 1] = add(lsf[i], 2 * GAP3);
        }
    }
    if lsf[M - 1] > M_LIMIT {
        lsf[M - 1] = M_LIMIT;
    }
    if lsf[M - 1] < lsf[M - 2] {
        lsf[M - 2] = sub(lsf[M - 1], GAP3);
    }

    let mut weight = [0i16; M];
    get_wegt(&lsf, &mut weight);

    let mut errlsf = [[0i16; M]; 2];
    for mode in 0..2 {
        lsp_prev_extract(
            &lsf,
            &mut errlsf[mode],
            &noise_fg[mode],
            &NOISE_FG_SUM_INV[mode],
            freq_prev,
        );
    }

    let (mode, clust, mut tmpbuf) = qnt_e(&errlsf, &weight);
    ana[0] = mode as i16;
    ana[1] = clust[0] as i16;
    ana[2] = clust[1] as i16;

    lsp_expand(&mut tmpbuf, 10, 1, M);
    let mut lsfq = [0i16; M];
    lsp_prev_compose(
        &tmpbuf,
        &mut lsfq,
        &noise_fg[mode],
        &NOISE_FG_SUM[mode],
        freq_prev,
    );
    lsp_prev_update(&tmpbuf, freq_prev);
    lsp_stability(&mut lsfq);
    lsf_lsp2(&lsfq, lspq);
}

/// The SID codevector for codebook indices `c0` (first stage) and `c1`.
fn sid_codevector(c0: usize, c1: usize) -> [i16; M] {
    let mut q = LSPCB1[PTRTAB_1[c0] as usize];
    for i in 0..M / 2 {
        q[i] = add(q[i], LSPCB2[PTRTAB_2[0][c1] as usize][i]);
    }
    for i in M / 2..M {
        q[i] = add(q[i], LSPCB2[PTRTAB_2[1][c1] as usize][i]);
    }
    q
}

/// Two-stage M-best search over both predictors. Returns the predictor,
/// the two codebook indices and the (unexpanded) codevector.
fn qnt_e(errlsf: &[[i16; M]; 2], weight: &[i16; M]) -> (usize, [usize; 2], [i16; M]) {
    const K1: usize = 4;
    let (d1, best1, back1) = ml_search_1(errlsf, K1, 32);
    let (best2, back2) = ml_search_2(&d1, weight, &back1, 16);
    let c1 = best2;
    let c0 = best1[back2];
    let mode = back1[back2];
    (mode, [c0, c1], sid_codevector(c0, c1))
}

/// The `k` smallest of `sum` (first occurrence wins ties), as indices.
fn k_best(sum: &mut [i16], k: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(k);
    for _ in 0..k {
        let mut min = MAX_16;
        let mut at = 0;
        for (i, &v) in sum.iter().enumerate() {
            if sub(v, min) < 0 {
                min = v;
                at = i;
            }
        }
        sum[at] = MAX_16;
        out.push(at);
    }
    out
}

/// First stage: for each predictor's residual, distance to each of the
/// `mq` first-stage SID codevectors.
fn ml_search_1(d: &[[i16; M]; 2], k: usize, mq: usize) -> (Vec<[i16; M]>, Vec<usize>, Vec<usize>) {
    let mut sum = vec![0i16; d.len() * mq];
    for (p, dp) in d.iter().enumerate() {
        for m in 0..mq {
            let cb = &LSPCB1[PTRTAB_1[m] as usize];
            let mut acc = 0;
            for l in 0..M {
                let tmp = sub(dp[l], cb[l]);
                acc = l_mac(acc, tmp, tmp);
            }
            sum[p * mq + m] = mult(extract_h(acc), MP[p]);
        }
    }
    let picks = k_best(&mut sum, k);
    let mut new_d = Vec::with_capacity(k);
    let mut best = Vec::with_capacity(k);
    let mut back = Vec::with_capacity(k);
    for at in picks {
        let (p, m) = (at / mq, at % mq);
        let cb = &LSPCB1[PTRTAB_1[m] as usize];
        let mut nd = [0i16; M];
        for l in 0..M {
            nd[l] = sub(d[p][l], cb[l]);
        }
        new_d.push(nd);
        best.push(m);
        back.push(p);
    }
    (new_d, best, back)
}

/// Second stage, weighted: the single best (candidate, codevector) pair.
fn ml_search_2(d: &[[i16; M]], weight: &[i16; M], ptr_prd: &[usize], mq: usize) -> (usize, usize) {
    let mut sum = vec![0i16; d.len() * mq];
    for (p, dp) in d.iter().enumerate() {
        let fg_sum = &NOISE_FG_SUM[ptr_prd[p]];
        for m in 0..mq {
            let mut acc = 0;
            for l in 0..M {
                let half = usize::from(l >= M / 2);
                let cb = &LSPCB2[PTRTAB_2[half][m] as usize];
                let mut tmp1 = extract_h(l_shl(l_mult(fg_sum[l], fg_sum[l]), 2));
                tmp1 = mult(tmp1, weight[l]);
                let tmp2 = sub(dp[l], cb[l]);
                tmp1 = extract_h(l_shl(l_mult(tmp1, tmp2), 3));
                acc = l_mac(acc, tmp1, tmp2);
            }
            sum[p * mq + m] = extract_h(acc);
        }
    }
    let at = k_best(&mut sum, 1)[0];
    (at % mq, at / mq)
}

/// Decode a SID frame's LSF indices (`index` = predictor, first, second).
fn sid_lsfq_decode(
    index: &[i16],
    lspq: &mut [i16; M],
    freq_prev: &mut FreqPrev,
    noise_fg: &[MaPredictor; 2],
) {
    let mode = index[0] as usize;
    let mut tmpbuf = sid_codevector(index[1] as usize, index[2] as usize);
    for j in 1..M {
        let mut acc = l_mult(tmpbuf[j - 1], 16384);
        acc = l_mac(acc, tmpbuf[j], -16384);
        acc = l_mac(acc, 10, 16384);
        let k = extract_h(acc);
        if k > 0 {
            tmpbuf[j - 1] = sub(tmpbuf[j - 1], k);
            tmpbuf[j] = add(tmpbuf[j], k);
        }
    }
    let mut lsfq = [0i16; M];
    lsp_prev_compose(
        &tmpbuf,
        &mut lsfq,
        &noise_fg[mode],
        &NOISE_FG_SUM[mode],
        freq_prev,
    );
    lsp_prev_update(&tmpbuf, freq_prev);
    lsp_stability(&mut lsfq);
    lsf_lsp2(&lsfq, lspq);
}

// ---------------------------------------------------------------------
// Comfort-noise excitation

fn gauss(seed: &mut Random) -> i16 {
    let mut acc = 0;
    for _ in 0..12 {
        acc = l_add(acc, l_deposit_l(seed.draw()));
    }
    extract_l(l_shr(acc, 7))
}

fn sqrt(num: i32) -> i16 {
    let mut rez: i16 = 0;
    let mut exp: i16 = 0x4000;
    for _ in 0..14 {
        let acc = l_mult(add(rez, exp), add(rez, exp));
        if l_sub(num, acc) >= 0 {
            rez = add(rez, exp);
        }
        exp = shr(exp, 1);
    }
    rez
}

/// Comfort-noise excitation for one frame at `exc[pos..pos + L_FRAME]`:
/// random adaptive and fixed codebook contributions plus Gaussian noise,
/// scaled to `cur_gain`. The encoder passes its taming state, which tracks
/// the excitation it would have produced.
fn calc_exc_rand(
    cur_gain: i16,
    exc: &mut [i16],
    pos: usize,
    seed: &mut Random,
    mut taming: Option<&mut Taming>,
) {
    if cur_gain == 0 {
        exc[pos..pos + L_FRAME].fill(0);
        if let Some(t) = taming {
            for _ in 0..2 {
                t.update(0, (L_SUBFR + 1) as i16);
            }
        }
        return;
    }

    for sf in 0..2 {
        let cur = pos + sf * L_SUBFR;
        let mut pos_p = [0usize; 4];
        let mut sign = [0i16; 4];

        let mut temp1 = seed.draw();
        let mut frac = sub(temp1 & 0x0003, 1);
        if frac == 2 {
            frac = 0;
        }
        temp1 = shr(temp1, 2);
        let t0 = add(temp1 & 0x003f, 40);
        temp1 = shr(temp1, 6);
        let mut temp2 = temp1 & 0x0007;
        pos_p[0] = add(shl(temp2, 2), temp2) as usize;
        temp1 = shr(temp1, 3);
        sign[0] = temp1 & 1;
        temp1 = shr(temp1, 1);
        temp2 = temp1 & 0x0007;
        temp2 = add(shl(temp2, 2), temp2);
        pos_p[1] = add(temp2, 1) as usize;
        temp1 = shr(temp1, 3);
        sign[1] = temp1 & 1;

        temp1 = seed.draw();
        temp2 = temp1 & 0x0007;
        temp2 = add(shl(temp2, 2), temp2);
        pos_p[2] = add(temp2, 2) as usize;
        temp1 = shr(temp1, 3);
        sign[2] = temp1 & 1;
        temp1 = shr(temp1, 1);
        temp2 = temp1 & 0x000f;
        let mut p3 = add(temp2 & 1, 3);
        temp2 = shr(temp2, 1) & 7;
        temp2 = add(shl(temp2, 2), temp2);
        p3 = add(p3, temp2);
        pos_p[3] = p3 as usize;
        temp1 = shr(temp1, 4);
        sign[3] = temp1 & 1;

        let mut gp = seed.draw() & 0x1fff;
        let gp2 = shl(gp, 1);

        // Gaussian excitation, normalised then scaled to the gain.
        let mut excg = [0i16; L_SUBFR];
        let mut acc = 0;
        for e in &mut excg {
            let g = gauss(seed);
            acc = l_mac(acc, g, g);
            *e = g;
        }
        acc = inv_sqrt(l_shr(acc, 1));
        let (hi, lo) = l_extract(acc);
        let t = add(cur_gain, mult_r(cur_gain, FRAC1));
        acc = mpy_32_16(hi, lo, t);
        let mut sh = norm_l(acc);
        let t = extract_h(l_shl(acc, sh));
        sh = sub(sh, 14);
        for e in &mut excg {
            *e = shr_r(mult_r(*e, t), sh);
        }

        // Adaptive codebook part plus the Gaussian part.
        pred_lt_3(exc, cur, t0, frac, L_SUBFR);
        let mut max = 0;
        for i in 0..L_SUBFR {
            let v = add(mult_r(exc[cur + i], gp2), excg[i]);
            exc[cur + i] = v;
            let a = abs_s(v);
            if sub(a, max) > 0 {
                max = a;
            }
        }
        sh = if max == 0 {
            0
        } else {
            sub(3, norm_s(max)).max(0)
        };
        let mut excs = [0i16; L_SUBFR];
        for i in 0..L_SUBFR {
            excs[i] = shr(exc[cur + i], sh);
        }
        let mut l_ener = 0;
        for &v in &excs {
            l_ener = l_mac(l_ener, v, v);
        }
        let mut inter_exc: i16 = 0;
        for i in 0..4 {
            let v = excs[pos_p[i]];
            inter_exc = if sign[i] == 0 {
                sub(inter_exc, v)
            } else {
                add(inter_exc, v)
            };
        }

        // Solve for the fixed-codebook gain that brings the energy to the
        // target: g^2 * 4 + 2 g * inter_exc + (ener - k) = 0.
        let mut acc = l_shr(l_mult(cur_gain, L_SUBFR as i16), 6);
        let t = extract_l(acc);
        let l_k = l_mult(cur_gain, t);
        let t = add(1, shl(sh, 1));
        acc = l_shr(l_k, t);
        acc = l_sub(acc, l_ener);
        inter_exc = shr(inter_exc, 1);
        acc = l_mac(acc, inter_exc, inter_exc);
        sh = add(sh, 1);

        if acc < 0 {
            // No real root: fall back to the Gaussian excitation alone.
            exc[cur..cur + L_SUBFR].copy_from_slice(&excg);
            let t1 = abs_s(excg[pos_p[0]]) | abs_s(excg[pos_p[1]]);
            let t2 = abs_s(excg[pos_p[2]]) | abs_s(excg[pos_p[3]]);
            sh = if (t1 | t2) & 0x4000 == 0 { 1 } else { 2 };
            inter_exc = 0;
            for i in 0..4 {
                let v = shr(excg[pos_p[i]], sh);
                inter_exc = if sign[i] == 0 {
                    sub(inter_exc, v)
                } else {
                    add(inter_exc, v)
                };
            }
            let (hi, lo) = l_extract(l_k);
            acc = mpy_32_16(hi, lo, K0);
            acc = l_shr(acc, sub(shl(sh, 1), 1));
            acc = l_mac(acc, inter_exc, inter_exc);
            gp = 0;
        }

        let temp2 = sqrt(acc);
        let mut x1 = sub(temp2, inter_exc);
        let x2 = negate(add(inter_exc, temp2));
        if sub(abs_s(x2), abs_s(x1)) < 0 {
            x1 = x2;
        }
        let mut g = shr_r(x1, sub(2, sh));
        if g >= 0 {
            if sub(g, G_MAX) > 0 {
                g = G_MAX;
            }
        } else if add(g, G_MAX) < 0 {
            g = negate(G_MAX);
        }

        for i in 0..4 {
            let j = cur + pos_p[i];
            exc[j] = if sign[i] != 0 {
                add(exc[j], g)
            } else {
                sub(exc[j], g)
            };
        }

        if let Some(t) = taming.as_deref_mut() {
            t.update(gp, t0);
        }
    }
}

// ---------------------------------------------------------------------
// Encoder

/// Shared state the encoder's comfort-noise generator works on.
pub struct CngContext<'a> {
    pub exc: &'a mut [i16],
    pub pos: usize,
    pub past_vad: bool,
    pub lsp_old_q: &'a mut [i16; M],
    pub aq: &'a mut [i16; 2 * MP1],
    pub freq_prev: &'a mut FreqPrev,
    pub seed: &'a mut Random,
    pub levinson: &'a mut Levinson,
    pub taming: &'a mut Taming,
}

pub struct CngEncoder {
    noise_fg: [MaPredictor; 2],
    lsp_sid_q: [i16; M],
    past_coeff: [i16; MP1],
    r_coeff: [i16; MP1],
    sh_r_coeff: i16,
    acf: [i16; SIZ_ACF],
    sh_acf: [i16; NB_CURACF],
    sum_acf: [i16; SIZ_SUMACF],
    sh_sum_acf: [i16; NB_SUMACF],
    ener: [i16; NB_GAIN],
    sh_ener: [i16; NB_GAIN],
    fr_cur: i16,
    cur_gain: i16,
    nb_ener: usize,
    sid_gain: i16,
    flag_chang: bool,
    prev_energy: i16,
    count_fr0: i16,
}

impl CngEncoder {
    pub fn new() -> Self {
        Self {
            noise_fg: noise_fg(),
            lsp_sid_q: [0; M],
            past_coeff: [0; MP1],
            r_coeff: [0; MP1],
            sh_r_coeff: 0,
            acf: [0; SIZ_ACF],
            sh_acf: [40; NB_CURACF],
            sum_acf: [0; SIZ_SUMACF],
            sh_sum_acf: [40; NB_SUMACF],
            ener: [0; NB_GAIN],
            sh_ener: [40; NB_GAIN],
            fr_cur: 0,
            cur_gain: 0,
            nb_ener: 0,
            sid_gain: 0,
            flag_chang: false,
            prev_energy: 0,
            count_fr0: 0,
        }
    }

    /// Record this frame's (un-windowed) autocorrelation, every frame.
    pub fn update(&mut self, r_h: &[i16], exp_r: i16, vad: bool) {
        self.acf.copy_within(0..SIZ_ACF - MP1, MP1);
        self.sh_acf.copy_within(0..NB_CURACF - 1, 1);
        self.sh_acf[0] = negate(add(16, exp_r));
        self.acf[..MP1].copy_from_slice(&r_h[..MP1]);
        self.fr_cur = add(self.fr_cur, 1);
        if self.fr_cur as usize == NB_CURACF {
            self.fr_cur = 0;
            if vad {
                self.update_sum_acf();
            }
        }
    }

    /// Encode an inactive frame: decide between a SID and sending nothing,
    /// and generate the comfort-noise excitation and LP filters the decoder
    /// will also use. Returns the SID parameters (predictor, two LSF
    /// indices, gain) when a SID is due.
    pub fn encode(&mut self, ctx: CngContext) -> Option<[i16; 4]> {
        self.ener.copy_within(0..NB_GAIN - 1, 1);
        self.sh_ener.copy_within(0..NB_GAIN - 1, 1);

        let (cur_acf, sh0) = calc_sum_acf(&self.acf, &self.sh_acf, NB_CURACF);
        self.sh_ener[0] = sh0;
        let mut cur_coeff = [0i16; MP1];
        if cur_acf[0] == 0 {
            self.ener[0] = 0;
        } else {
            let mut bid = [0i16; M];
            ctx.levinson.run(
                &cur_acf,
                &[0; MP1],
                &mut cur_coeff,
                &mut bid,
                &mut self.ener[0],
            );
        }

        let send_sid;
        let (energyq, cur_igain) = if ctx.past_vad {
            send_sid = true;
            self.count_fr0 = 0;
            self.nb_ener = 1;
            qua_sidgain(&self.ener, &self.sh_ener, self.nb_ener)
        } else {
            self.nb_ener = (self.nb_ener + 1).min(NB_GAIN);
            let q = qua_sidgain(&self.ener, &self.sh_ener, self.nb_ener);
            if cmp_filt(
                &self.r_coeff,
                self.sh_r_coeff,
                &cur_acf,
                self.ener[0],
                FRAC_THRESH1,
            ) {
                self.flag_chang = true;
            }
            if sub(abs_s(sub(self.prev_energy, q.0)), 2) > 0 {
                self.flag_chang = true;
            }
            self.count_fr0 = add(self.count_fr0, 1);
            if sub(self.count_fr0, FR_SID_MIN) < 0 {
                send_sid = false;
            } else {
                send_sid = self.flag_chang;
                self.count_fr0 = FR_SID_MIN;
            }
            q
        };

        let mut ana = [0i16; 4];
        if send_sid {
            self.count_fr0 = 0;
            self.flag_chang = false;
            self.past_coeff = self.calc_pastfilt(ctx.levinson);
            (self.r_coeff, self.sh_r_coeff) = calc_r_coeff(&self.past_coeff);
            let lpc = if cmp_filt(
                &self.r_coeff,
                self.sh_r_coeff,
                &cur_acf,
                self.ener[0],
                FRAC_THRESH2,
            ) {
                (self.r_coeff, self.sh_r_coeff) = calc_r_coeff(&cur_coeff);
                cur_coeff
            } else {
                self.past_coeff
            };
            let mut lsp_new = [0i16; M];
            az_lsp(&lpc, &mut lsp_new, ctx.lsp_old_q);
            lsfq_noise(
                &lsp_new,
                &mut self.lsp_sid_q,
                ctx.freq_prev,
                &self.noise_fg,
                &mut ana[..3],
            );
            self.prev_energy = energyq;
            ana[3] = cur_igain;
            self.sid_gain = TAB_SIDGAIN[cur_igain as usize];
        }

        if ctx.past_vad {
            self.cur_gain = self.sid_gain;
        } else {
            self.cur_gain = mult_r(self.cur_gain, A_GAIN0);
            self.cur_gain = add(self.cur_gain, mult_r(self.sid_gain, A_GAIN1));
        }
        calc_exc_rand(self.cur_gain, ctx.exc, ctx.pos, ctx.seed, Some(ctx.taming));
        int_qlpc(ctx.lsp_old_q, &self.lsp_sid_q, ctx.aq);
        *ctx.lsp_old_q = self.lsp_sid_q;

        if self.fr_cur == 0 {
            self.update_sum_acf();
        }

        send_sid.then_some(ana)
    }

    fn calc_pastfilt(&self, levinson: &mut Levinson) -> [i16; MP1] {
        let (s_sum_acf, _) = calc_sum_acf(&self.sum_acf, &self.sh_sum_acf, NB_SUMACF);
        let mut coeff = [0i16; MP1];
        if s_sum_acf[0] == 0 {
            coeff[0] = 4096;
            return coeff;
        }
        let mut bid = [0i16; M];
        let mut temp = 0;
        levinson.run(&s_sum_acf, &[0; MP1], &mut coeff, &mut bid, &mut temp);
        coeff
    }

    fn update_sum_acf(&mut self) {
        self.sum_acf.copy_within(0..SIZ_SUMACF - MP1, MP1);
        self.sh_sum_acf.copy_within(0..NB_SUMACF - 1, 1);
        let (sum, sh) = calc_sum_acf(&self.acf, &self.sh_acf, NB_CURACF);
        self.sum_acf[..MP1].copy_from_slice(&sum);
        self.sh_sum_acf[0] = sh;
    }
}

/// Autocorrelation of a filter's coefficients, normalised.
fn calc_r_coeff(coeff: &[i16; MP1]) -> ([i16; MP1], i16) {
    let mut acc = 0;
    for &c in coeff {
        acc = l_mac(acc, c, c);
    }
    let sh1 = norm_l(acc);
    let mut r = [0i16; MP1];
    r[0] = round(l_shl(acc, sh1));
    for i in 1..=M {
        let mut acc = 0;
        for j in 0..=M - i {
            acc = l_mac(acc, coeff[j], coeff[j + i]);
        }
        r[i] = round(l_shl(acc, sh1));
    }
    (r, sh1)
}

/// Whether the filter with autocorrelation `r_coeff` predicts a signal
/// with autocorrelation `acf` worse than `alpha * (1 + thresh)`, i.e. the
/// noise spectrum changed.
fn cmp_filt(
    r_coeff: &[i16; MP1],
    sh_r_coeff: i16,
    acf: &[i16; MP1],
    alpha: i16,
    thresh: i16,
) -> bool {
    let mut sh = [0i16; 2];
    let mut ind = 1;
    let l_temp0 = loop {
        clear_overflow();
        let mut acc = l_shr(l_mult(shr(r_coeff[0], sh[0]), shr(acf[0], sh[1])), 1);
        for i in 1..=M {
            acc = l_mac(acc, shr(r_coeff[i], sh[0]), shr(acf[i], sh[1]));
        }
        if !overflow() {
            break acc;
        }
        sh[ind] = add(sh[ind], 1);
        ind = 1 - ind;
    };
    let temp1 = mult_r(alpha, thresh);
    let mut l_temp1 = l_add(l_deposit_l(temp1), l_deposit_l(alpha));
    let shift = sub(add(sh_r_coeff, 9), add(sh[0], sh[1]));
    l_temp1 = l_shl(l_temp1, shift);
    l_sub(l_temp0, l_temp1) > 0
}

/// Sum `nb` autocorrelation vectors of differing scale; returns the sum
/// and its scale.
fn calc_sum_acf(acf: &[i16], sh_acf: &[i16], nb: usize) -> ([i16; MP1], i16) {
    let mut sh0 = sh_acf[0];
    for &s in &sh_acf[1..nb] {
        if sub(s, sh0) < 0 {
            sh0 = s;
        }
    }
    sh0 = add(sh0, 14);
    let mut l_tab = [0i32; MP1];
    for i in 0..nb {
        let temp = sub(sh0, sh_acf[i]);
        for j in 0..MP1 {
            let t = l_shl(l_deposit_l(acf[i * MP1 + j]), temp);
            l_tab[j] = l_add(l_tab[j], t);
        }
    }
    let temp = norm_l(l_tab[0]);
    let mut sum = [0i16; MP1];
    for i in 0..MP1 {
        sum[i] = extract_h(l_shl(l_tab[i], temp));
    }
    (sum, add(sh0, sub(temp, 16)))
}

// ---------------------------------------------------------------------
// Decoder

/// Shared state the decoder's comfort-noise generator works on.
pub struct CngDecodeContext<'a> {
    pub past_speech: bool,
    pub sid_sav: i16,
    pub sh_sid_sav: i16,
    /// The SID frame's parameters, when one arrived.
    pub sid: Option<&'a [i16; 4]>,
    pub exc: &'a mut [i16],
    pub pos: usize,
    pub lsp_old: &'a mut [i16; M],
    pub a_t: &'a mut [i16; 2 * MP1],
    pub seed: &'a mut Random,
    pub freq_prev: &'a mut FreqPrev,
}

pub struct CngDecoder {
    noise_fg: [MaPredictor; 2],
    cur_gain: i16,
    lsp_sid: [i16; M],
    sid_gain: i16,
}

impl CngDecoder {
    pub fn new() -> Self {
        Self {
            noise_fg: noise_fg(),
            cur_gain: 0,
            lsp_sid: [
                31441, 27566, 21458, 13612, 4663, -4663, -13612, -21458, -27566, -31441,
            ],
            sid_gain: TAB_SIDGAIN[0],
        }
    }

    pub fn decode(&mut self, ctx: CngDecodeContext) {
        if let Some(sid) = ctx.sid {
            self.sid_gain = TAB_SIDGAIN[sid[3] as usize];
            sid_lsfq_decode(&sid[..3], &mut self.lsp_sid, ctx.freq_prev, &self.noise_fg);
        } else if ctx.past_speech {
            let (_, ind) = qua_sidgain(&[ctx.sid_sav], &[ctx.sh_sid_sav], 0);
            self.sid_gain = TAB_SIDGAIN[ind as usize];
        }

        if ctx.past_speech {
            self.cur_gain = self.sid_gain;
        } else {
            self.cur_gain = mult_r(self.cur_gain, A_GAIN0);
            self.cur_gain = add(self.cur_gain, mult_r(self.sid_gain, A_GAIN1));
        }
        calc_exc_rand(self.cur_gain, ctx.exc, ctx.pos, ctx.seed, None);
        int_qlpc(ctx.lsp_old, &self.lsp_sid, ctx.a_t);
        *ctx.lsp_old = self.lsp_sid;
    }
}

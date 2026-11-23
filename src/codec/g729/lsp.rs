//! LSP vector quantisation: the encoder's two-stage, switched-MA-predictor
//! search and the decoder's reconstruction (QUA_LSP.C, LSPGETQ.C,
//! LSPDEC.C).

use super::basic_op::{
    MAX_32, add, extract_h, l_deposit_h, l_deposit_l, l_mac, l_msu, l_mult, l_shl, l_sub, mult,
    norm_s, shl, shr, sub,
};
use super::lpc::{lsf_lsp2, lsp_lsf2};
use super::tables::{FG, FG_SUM, FG_SUM_INV, LSPCB1, LSPCB2};
use super::{M, NC};

const MA_NP: usize = 4;
const MODE: usize = 2;
const NC0_B: i16 = 7;
const NC1_B: i16 = 5;
const NC1: usize = 1 << NC1_B;

const L_LIMIT: i16 = 40;
const M_LIMIT: i16 = 25681;
const GAP1: i16 = 10;
const GAP2: i16 = 5;
const GAP3: i16 = 321;
const PI04: i16 = 1029;
const PI92: i16 = 23677;
const CONST10: i16 = 10 * (1 << 11);
const CONST12: i16 = 19661;

const FREQ_PREV_RESET: [i16; M] = [
    2339, 4679, 7018, 9358, 11698, 14037, 16377, 18717, 21056, 23396,
];

type FreqPrev = [[i16; M]; MA_NP];

/// Push the codevector pairs in `buf[from..to]` at least `gap` apart.
fn lsp_expand(buf: &mut [i16; M], gap: i16, from: usize, to: usize) {
    for j in from..to {
        let diff = sub(buf[j - 1], buf[j]);
        let tmp = shr(add(diff, gap), 1);
        if tmp > 0 {
            buf[j - 1] = sub(buf[j - 1], tmp);
            buf[j] = add(buf[j], tmp);
        }
    }
}

fn lsp_prev_compose(lsp_ele: &[i16; M], lsp: &mut [i16; M], mode: usize, freq_prev: &FreqPrev) {
    for j in 0..M {
        let mut acc = l_mult(lsp_ele[j], FG_SUM[mode][j]);
        for k in 0..MA_NP {
            acc = l_mac(acc, freq_prev[k][j], FG[mode][k][j]);
        }
        lsp[j] = extract_h(acc);
    }
}

fn lsp_prev_extract(lsp: &[i16; M], lsp_ele: &mut [i16; M], mode: usize, freq_prev: &FreqPrev) {
    for j in 0..M {
        let mut t = l_deposit_h(lsp[j]);
        for k in 0..MA_NP {
            t = l_msu(t, freq_prev[k][j], FG[mode][k][j]);
        }
        let temp = extract_h(t);
        t = l_mult(temp, FG_SUM_INV[mode][j]);
        lsp_ele[j] = extract_h(l_shl(t, 3));
    }
}

fn lsp_prev_update(lsp_ele: &[i16; M], freq_prev: &mut FreqPrev) {
    freq_prev.copy_within(0..MA_NP - 1, 1);
    freq_prev[0] = *lsp_ele;
}

fn lsp_stability(buf: &mut [i16; M]) {
    for j in 0..M - 1 {
        if l_sub(l_deposit_l(buf[j + 1]), l_deposit_l(buf[j])) < 0 {
            buf.swap(j, j + 1);
        }
    }
    if sub(buf[0], L_LIMIT) < 0 {
        buf[0] = L_LIMIT;
    }
    for j in 0..M - 1 {
        let diff = l_sub(l_deposit_l(buf[j + 1]), l_deposit_l(buf[j]));
        if l_sub(diff, i32::from(GAP3)) < 0 {
            buf[j + 1] = add(buf[j], GAP3);
        }
    }
    if sub(buf[M - 1], M_LIMIT) > 0 {
        buf[M - 1] = M_LIMIT;
    }
}

fn lsp_get_quant(
    code0: usize,
    code1: usize,
    code2: usize,
    mode: usize,
    freq_prev: &mut FreqPrev,
    lspq: &mut [i16; M],
) {
    let mut buf = [0i16; M];
    for j in 0..NC {
        buf[j] = add(LSPCB1[code0][j], LSPCB2[code1][j]);
    }
    for j in NC..M {
        buf[j] = add(LSPCB1[code0][j], LSPCB2[code2][j]);
    }
    lsp_expand(&mut buf, GAP1, 1, M);
    lsp_expand(&mut buf, GAP2, 1, M);
    lsp_prev_compose(&buf, lspq, mode, freq_prev);
    lsp_prev_update(&buf, freq_prev);
    lsp_stability(lspq);
}

/// Perceptual weights for the LSF error.
fn get_wegt(flsp: &[i16; M], wegt: &mut [i16; M]) {
    let mut buf = [0i16; M];
    buf[0] = sub(flsp[1], PI04 + 8192);
    for i in 1..M - 1 {
        buf[i] = sub(sub(flsp[i + 1], flsp[i - 1]), 8192);
    }
    buf[M - 1] = sub(PI92 - 8192, flsp[M - 2]);

    for i in 0..M {
        wegt[i] = if buf[i] > 0 {
            2048
        } else {
            let mut acc = l_mult(buf[i], buf[i]);
            let tmp = extract_h(l_shl(acc, 2));
            acc = l_mult(tmp, CONST10);
            add(extract_h(l_shl(acc, 2)), 2048)
        };
    }
    wegt[4] = extract_h(l_shl(l_mult(wegt[4], CONST12), 1));
    wegt[5] = extract_h(l_shl(l_mult(wegt[5], CONST12), 1));

    let mut max = 0;
    for &w in wegt.iter() {
        if sub(w, max) > 0 {
            max = w;
        }
    }
    let sft = norm_s(max);
    for w in wegt.iter_mut() {
        *w = shl(*w, sft);
    }
}

fn lsp_pre_select(rbuf: &[i16; M]) -> usize {
    let mut cand = 0;
    let mut dmin = MAX_32;
    for (i, cb) in LSPCB1.iter().enumerate() {
        let mut t = 0;
        for j in 0..M {
            let d = sub(rbuf[j], cb[j]);
            t = l_mac(t, d, d);
        }
        if l_sub(t, dmin) < 0 {
            dmin = t;
            cand = i;
        }
    }
    cand
}

/// Best second-stage vector for `rbuf[range] - cb1[range]`.
fn lsp_select(
    rbuf: &[i16; M],
    cb1: &[i16; M],
    wegt: &[i16; M],
    range: std::ops::Range<usize>,
) -> usize {
    let mut buf = [0i16; M];
    for j in range.clone() {
        buf[j] = sub(rbuf[j], cb1[j]);
    }
    let mut index = 0;
    let mut dmin = MAX_32;
    for (k, cb2) in LSPCB2.iter().enumerate().take(NC1) {
        let mut dist = 0;
        for j in range.clone() {
            let tmp = sub(buf[j], cb2[j]);
            let tmp2 = mult(wegt[j], tmp);
            dist = l_mac(dist, tmp2, tmp);
        }
        if l_sub(dist, dmin) < 0 {
            dmin = dist;
            index = k;
        }
    }
    index
}

fn lsp_get_tdist(wegt: &[i16; M], buf: &[i16; M], rbuf: &[i16; M], mode: usize) -> i32 {
    let mut tdist = 0;
    for j in 0..M {
        let tmp = mult(sub(buf[j], rbuf[j]), FG_SUM[mode][j]);
        let acc = l_mult(wegt[j], tmp);
        let tmp2 = extract_h(l_shl(acc, 4));
        tdist = l_mac(tdist, tmp2, tmp);
    }
    tdist
}

/// Encoder LSP quantiser state (the MA predictor memory).
pub struct LspEncoder {
    freq_prev: FreqPrev,
}

impl LspEncoder {
    pub const fn new() -> Self {
        Self {
            freq_prev: [FREQ_PREV_RESET; MA_NP],
        }
    }

    /// Quantise `lsp` into `lsp_q`, writing the two LSP indices to `ana`.
    pub fn quantize(&mut self, lsp: &[i16; M], lsp_q: &mut [i16; M], ana: &mut [i16]) {
        let mut lsf = [0i16; M];
        let mut lsf_q = [0i16; M];
        lsp_lsf2(lsp, &mut lsf);

        let mut wegt = [0i16; M];
        get_wegt(&lsf, &mut wegt);

        let mut cand = [0usize; MODE];
        let mut tindex1 = [0usize; MODE];
        let mut tindex2 = [0usize; MODE];
        let mut tdist = [0i32; MODE];
        let mut rbuf = [0i16; M];
        let mut buf = [0i16; M];

        for mode in 0..MODE {
            lsp_prev_extract(&lsf, &mut rbuf, mode, &self.freq_prev);
            let c = lsp_pre_select(&rbuf);
            cand[mode] = c;
            let cb1 = &LSPCB1[c];

            let index = lsp_select(&rbuf, cb1, &wegt, 0..NC);
            tindex1[mode] = index;
            for j in 0..NC {
                buf[j] = add(cb1[j], LSPCB2[index][j]);
            }
            lsp_expand(&mut buf, GAP1, 1, NC);

            let index = lsp_select(&rbuf, cb1, &wegt, NC..M);
            tindex2[mode] = index;
            for j in NC..M {
                buf[j] = add(cb1[j], LSPCB2[index][j]);
            }
            lsp_expand(&mut buf, GAP1, NC, M);
            lsp_expand(&mut buf, GAP2, 1, M);

            tdist[mode] = lsp_get_tdist(&wegt, &buf, &rbuf, mode);
        }

        let mode = usize::from(l_sub(tdist[1], tdist[0]) < 0);
        ana[0] = shl(mode as i16, NC0_B) | cand[mode] as i16;
        ana[1] = shl(tindex1[mode] as i16, NC1_B) | tindex2[mode] as i16;

        lsp_get_quant(
            cand[mode],
            tindex1[mode],
            tindex2[mode],
            mode,
            &mut self.freq_prev,
            &mut lsf_q,
        );
        lsf_lsp2(&lsf_q, lsp_q);
    }
}

/// Decoder LSP state: the MA predictor memory and the last good LSFs.
pub struct LspDecoder {
    freq_prev: FreqPrev,
    prev_ma: usize,
    prev_lsp: [i16; M],
}

impl LspDecoder {
    pub const fn new() -> Self {
        Self {
            freq_prev: [FREQ_PREV_RESET; MA_NP],
            prev_ma: 0,
            prev_lsp: FREQ_PREV_RESET,
        }
    }

    /// Decode the two LSP indices into `lsp_q`; on an erased frame repeat
    /// the last LSFs.
    pub fn decode(&mut self, prm: &[i16], lsp_q: &mut [i16; M], erase: bool) {
        let mut lsf_q = [0i16; M];
        if erase {
            lsf_q = self.prev_lsp;
            let mut buf = [0i16; M];
            lsp_prev_extract(&self.prev_lsp, &mut buf, self.prev_ma, &self.freq_prev);
            lsp_prev_update(&buf, &mut self.freq_prev);
        } else {
            let mode = (shr(prm[0], NC0_B) & 1) as usize;
            let code0 = (prm[0] & ((1 << NC0_B) - 1)) as usize;
            let code1 = (shr(prm[1], NC1_B) & (NC1 as i16 - 1)) as usize;
            let code2 = (prm[1] & (NC1 as i16 - 1)) as usize;
            lsp_get_quant(code0, code1, code2, mode, &mut self.freq_prev, &mut lsf_q);
            self.prev_lsp = lsf_q;
            self.prev_ma = mode;
        }
        lsf_lsp2(&lsf_q, lsp_q);
    }
}

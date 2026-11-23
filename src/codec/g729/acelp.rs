//! Algebraic (fixed) codebook: the Annex A fast search and its decoding
//! (ACELP_CA.C, DE_ACELP.C, COR_FUNC.C).
//!
//! The search works on a 616-entry matrix of impulse-response
//! correlations laid out exactly as the reference lays it out, so the
//! pointer walks of the C code become offsets into one array.

use super::L_SUBFR;
use super::basic_op::{
    MAX_16, MIN_16, add, extract_h, extract_l, l_abs, l_mac, l_msu, l_mult, l_shr, l_sub, mult,
    negate, norm_l, round, shl, shr, sub,
};

const NB_POS: usize = 8;
const STEP: usize = 5;
const MSIZE: usize = 64;
const DIM_RR: usize = 616;

const HALF: i16 = 16384;
const QUARTER: i16 = 8192;
const EIGHTH: i16 = 4096;
const SIXTEENTH: i16 = 2048;

// Offsets of the sub-matrices inside `rr`.
const RRI0I0: usize = 0;
const RRI1I1: usize = RRI0I0 + NB_POS;
const RRI2I2: usize = RRI1I1 + NB_POS;
const RRI3I3: usize = RRI2I2 + NB_POS;
const RRI4I4: usize = RRI3I3 + NB_POS;
const RRI0I1: usize = RRI4I4 + NB_POS;
const RRI0I2: usize = RRI0I1 + MSIZE;
const RRI0I3: usize = RRI0I2 + MSIZE;
const RRI0I4: usize = RRI0I3 + MSIZE;
const RRI1I2: usize = RRI0I4 + MSIZE;
const RRI1I3: usize = RRI1I2 + MSIZE;
const RRI1I4: usize = RRI1I3 + MSIZE;
const RRI2I3: usize = RRI1I4 + MSIZE;
const RRI2I4: usize = RRI2I3 + MSIZE;

/// Correlation of the target `x` with the impulse response `h`,
/// normalised to 13 bits.
pub fn cor_h_x(h: &[i16], target: &[i16], dn: &mut [i16; L_SUBFR]) {
    let mut y32 = [0i32; L_SUBFR];
    let mut max = 0;
    for i in 0..L_SUBFR {
        let mut acc = 0;
        for j in i..L_SUBFR {
            acc = l_mac(acc, target[j], h[j - i]);
        }
        y32[i] = acc;
        let mag = l_abs(acc);
        if l_sub(mag, max) > 0 {
            max = mag;
        }
    }
    let shift = sub(18, norm_l(max).min(16));
    for (out, &v) in dn.iter_mut().zip(&y32) {
        *out = extract_l(l_shr(v, shift));
    }
}

/// Fill `rr` with the correlations of the (rescaled) impulse response.
fn cor_h(hin: &[i16], rr: &mut [i16; DIM_RR]) {
    let mut h = [0i16; L_SUBFR];
    let mut cor = 0;
    for &v in &hin[..L_SUBFR] {
        cor = l_mac(cor, v, v);
    }
    if sub(extract_h(cor), 32000) > 0 {
        for (o, &v) in h.iter_mut().zip(hin) {
            *o = shr(v, 1);
        }
    } else {
        let k = shr(norm_l(cor), 1);
        for (o, &v) in h.iter_mut().zip(hin) {
            *o = shl(v, k);
        }
    }

    // Diagonals.
    let mut cor = 0;
    let mut hp = 0;
    for i in 0..NB_POS {
        let at = NB_POS - 1 - i;
        for base in [RRI4I4, RRI3I3, RRI2I2, RRI1I1, RRI0I0] {
            cor = l_mac(cor, h[hp], h[hp]);
            hp += 1;
            rr[base + at] = extract_h(cor);
        }
    }

    // Off-diagonals. Each block walks `h` against a shifted copy of itself
    // and drops the running correlation into up to five sub-matrices,
    // stepping back `NB_POS + 1` entries per row. `stores` lists, for each
    // of the five products in a row, which pointer (if any) takes it.
    let ldec = (NB_POS + 1) as isize;
    let set = |rr: &mut [i16; DIM_RR], p: isize, v: i16| rr[p as usize] = v;

    // Block 1: i0/i1, i1/i2, i2/i3, i0/i4.
    {
        let mut l_fin_sup = (MSIZE - 1) as isize;
        let mut l_fin_inf = l_fin_sup - 1;
        let mut hf = 1;
        for k in 0..NB_POS {
            let mut p3 = RRI2I3 as isize + l_fin_sup;
            let mut p2 = RRI1I2 as isize + l_fin_sup;
            let mut p1 = RRI0I1 as isize + l_fin_sup;
            let mut p0 = RRI0I4 as isize + l_fin_inf;
            let mut cor = 0;
            let (mut h1, mut h2) = (0, hf);
            let mut mac = |cor: &mut i32| {
                *cor = l_mac(*cor, h[h1], h[h2]);
                h1 += 1;
                h2 += 1;
            };
            for _ in k + 1..NB_POS {
                mac(&mut cor);
                mac(&mut cor);
                set(rr, p3, extract_h(cor));
                mac(&mut cor);
                set(rr, p2, extract_h(cor));
                mac(&mut cor);
                set(rr, p1, extract_h(cor));
                mac(&mut cor);
                set(rr, p0, extract_h(cor));
                p3 -= ldec;
                p2 -= ldec;
                p1 -= ldec;
                p0 -= ldec;
            }
            mac(&mut cor);
            mac(&mut cor);
            set(rr, p3, extract_h(cor));
            mac(&mut cor);
            set(rr, p2, extract_h(cor));
            mac(&mut cor);
            set(rr, p1, extract_h(cor));
            l_fin_sup -= NB_POS as isize;
            l_fin_inf -= 1;
            hf += STEP;
        }
    }

    // Block 2: i2/i4, i1/i3, i0/i2, i1/i4, i0/i3.
    {
        let mut l_fin_sup = (MSIZE - 1) as isize;
        let mut l_fin_inf = l_fin_sup - 1;
        let mut hf = 2;
        for k in 0..NB_POS {
            let mut p4 = RRI2I4 as isize + l_fin_sup;
            let mut p3 = RRI1I3 as isize + l_fin_sup;
            let mut p2 = RRI0I2 as isize + l_fin_sup;
            let mut p1 = RRI1I4 as isize + l_fin_inf;
            let mut p0 = RRI0I3 as isize + l_fin_inf;
            let mut cor = 0;
            let (mut h1, mut h2) = (0, hf);
            let mut mac = |cor: &mut i32| {
                *cor = l_mac(*cor, h[h1], h[h2]);
                h1 += 1;
                h2 += 1;
            };
            for _ in k + 1..NB_POS {
                mac(&mut cor);
                set(rr, p4, extract_h(cor));
                mac(&mut cor);
                set(rr, p3, extract_h(cor));
                mac(&mut cor);
                set(rr, p2, extract_h(cor));
                mac(&mut cor);
                set(rr, p1, extract_h(cor));
                mac(&mut cor);
                set(rr, p0, extract_h(cor));
                p4 -= ldec;
                p3 -= ldec;
                p2 -= ldec;
                p1 -= ldec;
                p0 -= ldec;
            }
            mac(&mut cor);
            set(rr, p4, extract_h(cor));
            mac(&mut cor);
            set(rr, p3, extract_h(cor));
            mac(&mut cor);
            set(rr, p2, extract_h(cor));
            l_fin_sup -= NB_POS as isize;
            l_fin_inf -= 1;
            hf += STEP;
        }
    }

    // Block 3: i1/i4, i0/i3, i2/i4, i1/i3, i0/i2.
    {
        let mut l_fin_sup = (MSIZE - 1) as isize;
        let mut l_fin_inf = l_fin_sup - 1;
        let mut hf = 3;
        for k in 0..NB_POS {
            let mut p4 = RRI1I4 as isize + l_fin_sup;
            let mut p3 = RRI0I3 as isize + l_fin_sup;
            let mut p2 = RRI2I4 as isize + l_fin_inf;
            let mut p1 = RRI1I3 as isize + l_fin_inf;
            let mut p0 = RRI0I2 as isize + l_fin_inf;
            let mut cor = 0;
            let (mut h1, mut h2) = (0, hf);
            let mut mac = |cor: &mut i32| {
                *cor = l_mac(*cor, h[h1], h[h2]);
                h1 += 1;
                h2 += 1;
            };
            for _ in k + 1..NB_POS {
                mac(&mut cor);
                set(rr, p4, extract_h(cor));
                mac(&mut cor);
                set(rr, p3, extract_h(cor));
                mac(&mut cor);
                set(rr, p2, extract_h(cor));
                mac(&mut cor);
                set(rr, p1, extract_h(cor));
                mac(&mut cor);
                set(rr, p0, extract_h(cor));
                p4 -= ldec;
                p3 -= ldec;
                p2 -= ldec;
                p1 -= ldec;
                p0 -= ldec;
            }
            mac(&mut cor);
            set(rr, p4, extract_h(cor));
            mac(&mut cor);
            set(rr, p3, extract_h(cor));
            l_fin_sup -= NB_POS as isize;
            l_fin_inf -= 1;
            hf += STEP;
        }
    }

    // Block 4: i0/i4, i2/i3, i1/i2, i0/i1.
    {
        let mut l_fin_sup = (MSIZE - 1) as isize;
        let mut l_fin_inf = l_fin_sup - 1;
        let mut hf = 4;
        for k in 0..NB_POS {
            let mut p3 = RRI0I4 as isize + l_fin_sup;
            let mut p2 = RRI2I3 as isize + l_fin_inf;
            let mut p1 = RRI1I2 as isize + l_fin_inf;
            let mut p0 = RRI0I1 as isize + l_fin_inf;
            let mut cor = 0;
            let (mut h1, mut h2) = (0, hf);
            let mut mac = |cor: &mut i32| {
                *cor = l_mac(*cor, h[h1], h[h2]);
                h1 += 1;
                h2 += 1;
            };
            for _ in k + 1..NB_POS {
                mac(&mut cor);
                set(rr, p3, extract_h(cor));
                mac(&mut cor);
                mac(&mut cor);
                set(rr, p2, extract_h(cor));
                mac(&mut cor);
                set(rr, p1, extract_h(cor));
                mac(&mut cor);
                set(rr, p0, extract_h(cor));
                p3 -= ldec;
                p2 -= ldec;
                p1 -= ldec;
                p0 -= ldec;
            }
            mac(&mut cor);
            set(rr, p3, extract_h(cor));
            l_fin_sup -= NB_POS as isize;
            l_fin_inf -= 1;
            hf += STEP;
        }
    }
}

/// `i / 5` for a pulse position, as the reference computes it.
#[inline]
fn pos5(i: usize) -> usize {
    mult(i as i16, 6554) as usize
}

/// Annex A's depth-first search of the 17-bit, 4-pulse codebook.
fn d4i40_17_fast(
    dn: &mut [i16; L_SUBFR],
    rr: &mut [i16; DIM_RR],
    h: &[i16],
    cod: &mut [i16],
    y: &mut [i16],
    sign: &mut i16,
) -> i16 {
    let mut sign_dn = [0i16; L_SUBFR];
    let mut sign_dn_inv = [0i16; L_SUBFR];
    for i in 0..L_SUBFR {
        if dn[i] >= 0 {
            sign_dn[i] = MAX_16;
            sign_dn_inv[i] = MIN_16;
        } else {
            sign_dn[i] = MIN_16;
            sign_dn_inv[i] = MAX_16;
            dn[i] = negate(dn[i]);
        }
    }

    // Fold the signs of the backward-filtered target into the matrix.
    let (mut p0, mut p1, mut p2, mut p3) = (RRI0I1, RRI0I2, RRI0I3, RRI0I4);
    for i0 in (0..L_SUBFR).step_by(STEP) {
        let ps = if sign_dn[i0] < 0 {
            &sign_dn_inv
        } else {
            &sign_dn
        };
        for i1 in (1..L_SUBFR).step_by(STEP) {
            rr[p0] = mult(rr[p0], ps[i1]);
            rr[p1] = mult(rr[p1], ps[i1 + 1]);
            rr[p2] = mult(rr[p2], ps[i1 + 2]);
            rr[p3] = mult(rr[p3], ps[i1 + 3]);
            p0 += 1;
            p1 += 1;
            p2 += 1;
            p3 += 1;
        }
    }
    let (mut p0, mut p1, mut p2) = (RRI1I2, RRI1I3, RRI1I4);
    for i1 in (1..L_SUBFR).step_by(STEP) {
        let ps = if sign_dn[i1] < 0 {
            &sign_dn_inv
        } else {
            &sign_dn
        };
        for i2 in (2..L_SUBFR).step_by(STEP) {
            rr[p0] = mult(rr[p0], ps[i2]);
            rr[p1] = mult(rr[p1], ps[i2 + 1]);
            rr[p2] = mult(rr[p2], ps[i2 + 2]);
            p0 += 1;
            p1 += 1;
            p2 += 1;
        }
    }
    let (mut p0, mut p1) = (RRI2I3, RRI2I4);
    for i2 in (2..L_SUBFR).step_by(STEP) {
        let ps = if sign_dn[i2] < 0 {
            &sign_dn_inv
        } else {
            &sign_dn
        };
        for i3 in (3..L_SUBFR).step_by(STEP) {
            rr[p0] = mult(rr[p0], ps[i3]);
            rr[p1] = mult(rr[p1], ps[i3 + 1]);
            p0 += 1;
            p1 += 1;
        }
    }

    let mut psk: i16 = -1;
    let mut alpk: i16 = 1;
    let mut rri0i3_i4 = RRI0I3;
    let mut rri1i3_i4 = RRI1I3;
    let mut rri2i3_i4 = RRI2I3;
    let mut rri3i3_i4 = RRI3I3;
    let (mut ip0, mut ip1, mut ip2, mut ip3) = (0usize, 1usize, 2usize, 3usize);
    let (mut ix, mut iy) = (0usize, 0usize);
    let mut ps: i16 = 0;
    let mut tmp_vect = [0i16; NB_POS];

    for track in 3..5 {
        // Pulses i2 and i3 (track 3 or 4) first, from the two best i2.
        let mut sq: i16 = -1;
        let mut alp: i16 = 1;
        let mut prev_i0: Option<usize> = None;
        let mut i0 = 0;
        for _ in 0..2 {
            let mut max: i16 = -1;
            for j in (2..L_SUBFR).step_by(STEP) {
                if sub(dn[j], max) > 0 && prev_i0 != Some(j) {
                    max = dn[j];
                    i0 = j;
                }
            }
            prev_i0 = Some(i0);
            let j = pos5(i0);
            let ps1 = dn[i0];
            let alp1 = l_mult(rr[RRI2I2 + j], QUARTER);
            for (k, i1) in (track..L_SUBFR).step_by(STEP).enumerate() {
                let ps2 = add(ps1, dn[i1]);
                let mut alp2 = l_mac(alp1, rr[rri2i3_i4 + (j << 3) + k], HALF);
                alp2 = l_mac(alp2, rr[rri3i3_i4 + k], QUARTER);
                let sq2 = mult(ps2, ps2);
                let alp_16 = round(alp2);
                if l_msu(l_mult(alp, sq2), sq, alp_16) > 0 {
                    sq = sq2;
                    ps = ps2;
                    alp = alp_16;
                    ix = i0;
                    iy = i1;
                }
            }
        }

        let i0 = ix;
        let i1 = iy;
        let ps0 = ps;
        let alp0 = l_mult(alp, QUARTER);
        let mut sq: i16 = -1;
        let mut alp: i16 = 1;

        // Then pulses i0 and i1.
        for (k, t) in tmp_vect.iter_mut().enumerate() {
            let mut s = l_mult(rr[RRI1I2 + pos5(i0) + k * NB_POS], QUARTER);
            s = l_mac(s, rr[rri1i3_i4 + pos5(i1) + k * NB_POS], QUARTER);
            s = l_mac(s, rr[RRI1I1 + k], EIGHTH);
            *t = round(s);
        }

        for (a, i2) in (0..L_SUBFR).step_by(STEP).enumerate() {
            let ps1 = add(ps0, dn[i2]);
            let mut alp1 = l_mac(alp0, rr[RRI0I2 + pos5(i0) + a * NB_POS], EIGHTH);
            alp1 = l_mac(alp1, rr[rri0i3_i4 + pos5(i1) + a * NB_POS], EIGHTH);
            alp1 = l_mac(alp1, rr[RRI0I0 + a], SIXTEENTH);
            for (k, i3) in (1..L_SUBFR).step_by(STEP).enumerate() {
                let ps2 = add(ps1, dn[i3]);
                let mut alp2 = l_mac(alp1, rr[RRI0I1 + a * NB_POS + k], EIGHTH);
                alp2 = l_mac(alp2, tmp_vect[k], HALF);
                let sq2 = mult(ps2, ps2);
                let alp_16 = round(alp2);
                if l_msu(l_mult(alp, sq2), sq, alp_16) > 0 {
                    sq = sq2;
                    alp = alp_16;
                    ix = i2;
                    iy = i3;
                }
            }
        }

        if l_msu(l_mult(alpk, sq), psk, alp) > 0 {
            psk = sq;
            alpk = alp;
            ip2 = i0;
            ip3 = i1;
            ip0 = ix;
            ip1 = iy;
        }

        // Second ordering: pulse i3 (track 3 or 4) and i0 first.
        let mut sq: i16 = -1;
        let mut alp: i16 = 1;
        let mut prev_i0: Option<usize> = None;
        let mut i0 = 0;
        for _ in 0..2 {
            let mut max: i16 = -1;
            for j in (track..L_SUBFR).step_by(STEP) {
                if sub(dn[j], max) > 0 && prev_i0 != Some(j) {
                    max = dn[j];
                    i0 = j;
                }
            }
            prev_i0 = Some(i0);
            let j = pos5(i0);
            let ps1 = dn[i0];
            let alp1 = l_mult(rr[rri3i3_i4 + j], QUARTER);
            for (k, i1) in (0..L_SUBFR).step_by(STEP).enumerate() {
                let ps2 = add(ps1, dn[i1]);
                let mut alp2 = l_mac(alp1, rr[rri0i3_i4 + j + k * NB_POS], HALF);
                alp2 = l_mac(alp2, rr[RRI0I0 + k], QUARTER);
                let sq2 = mult(ps2, ps2);
                let alp_16 = round(alp2);
                if l_msu(l_mult(alp, sq2), sq, alp_16) > 0 {
                    sq = sq2;
                    ps = ps2;
                    alp = alp_16;
                    ix = i0;
                    iy = i1;
                }
            }
        }

        let i0 = ix;
        let i1 = iy;
        let i1_offset = pos5(i1) << 3;
        let ps0 = ps;
        let alp0 = l_mult(alp, QUARTER);
        let mut sq: i16 = -1;
        let mut alp: i16 = 1;

        for (k, t) in tmp_vect.iter_mut().enumerate() {
            let mut s = l_mult(rr[rri2i3_i4 + pos5(i0) + k * NB_POS], QUARTER);
            s = l_mac(s, rr[RRI0I2 + i1_offset + k], QUARTER);
            s = l_mac(s, rr[RRI2I2 + k], EIGHTH);
            *t = round(s);
        }

        for (a, i2) in (1..L_SUBFR).step_by(STEP).enumerate() {
            let ps1 = add(ps0, dn[i2]);
            let mut alp1 = l_mac(alp0, rr[rri1i3_i4 + pos5(i0) + a * NB_POS], EIGHTH);
            alp1 = l_mac(alp1, rr[RRI0I1 + i1_offset + a], EIGHTH);
            alp1 = l_mac(alp1, rr[RRI1I1 + a], SIXTEENTH);
            for (k, i3) in (2..L_SUBFR).step_by(STEP).enumerate() {
                let ps2 = add(ps1, dn[i3]);
                let mut alp2 = l_mac(alp1, rr[RRI1I2 + a * NB_POS + k], EIGHTH);
                alp2 = l_mac(alp2, tmp_vect[k], HALF);
                let sq2 = mult(ps2, ps2);
                let alp_16 = round(alp2);
                if l_msu(l_mult(alp, sq2), sq, alp_16) > 0 {
                    sq = sq2;
                    alp = alp_16;
                    ix = i2;
                    iy = i3;
                }
            }
        }

        if l_msu(l_mult(alpk, sq), psk, alp) > 0 {
            psk = sq;
            alpk = alp;
            ip3 = i0;
            ip0 = i1;
            ip1 = ix;
            ip2 = iy;
        }

        rri0i3_i4 = RRI0I4;
        rri1i3_i4 = RRI1I4;
        rri2i3_i4 = RRI2I4;
        rri3i3_i4 = RRI4I4;
    }

    let signs = [sign_dn[ip0], sign_dn[ip1], sign_dn[ip2], sign_dn[ip3]];
    let pos = [ip0, ip1, ip2, ip3];

    cod[..L_SUBFR].fill(0);
    for (&p, &s) in pos.iter().zip(&signs) {
        cod[p] = shr(s, 2);
    }

    y[..ip0].fill(0);
    for (i, j) in (ip0..L_SUBFR).zip(0..) {
        y[i] = if signs[0] > 0 { h[j] } else { negate(h[j]) };
    }
    for (&p, &s) in pos[1..].iter().zip(&signs[1..]) {
        for (i, j) in (p..L_SUBFR).zip(0..) {
            y[i] = if s > 0 {
                add(y[i], h[j])
            } else {
                sub(y[i], h[j])
            };
        }
    }

    let mut sgn = 0;
    for (bit, &s) in signs.iter().enumerate() {
        if s > 0 {
            sgn = add(sgn, 1 << bit);
        }
    }
    *sign = sgn;

    let ip0 = pos5(ip0) as i16;
    let ip1 = pos5(ip1) as i16;
    let ip2 = pos5(ip2) as i16;
    let i = pos5(ip3) as i16;
    let j = add(i, shl(i, 2));
    let j = sub(ip3 as i16, add(j, 3));
    let ip3 = add(shl(i, 1), j);
    let mut index = add(ip0, shl(ip1, 3));
    index = add(index, shl(ip2, 6));
    add(index, shl(ip3, 9))
}

/// Fixed-codebook search for one subframe. Returns the pulse-position
/// index and sets `sign`; `h` is sharpened in place by the pitch.
pub fn acelp_code_a(
    x: &[i16],
    h: &mut [i16; L_SUBFR],
    t0: i16,
    pitch_sharp: i16,
    code: &mut [i16; L_SUBFR],
    y: &mut [i16; L_SUBFR],
    sign: &mut i16,
) -> i16 {
    let sharp = shl(pitch_sharp, 1);
    let t0u = t0 as usize;
    if t0u < L_SUBFR {
        for i in t0u..L_SUBFR {
            h[i] = add(h[i], mult(h[i - t0u], sharp));
        }
    }
    let mut rr = [0i16; DIM_RR];
    cor_h(h, &mut rr);
    let mut dn = [0i16; L_SUBFR];
    cor_h_x(h, x, &mut dn);
    let index = d4i40_17_fast(&mut dn, &mut rr, h, code, y, sign);
    if t0u < L_SUBFR {
        for i in t0u..L_SUBFR {
            code[i] = add(code[i], mult(code[i - t0u], sharp));
        }
    }
    index
}

/// Rebuild the four-pulse codevector (Q13) from its index and signs.
pub fn decod_acelp(mut sign: i16, mut index: i16, cod: &mut [i16; L_SUBFR]) {
    let mut pos = [0i16; 4];
    let i = index & 7;
    pos[0] = add(i, shl(i, 2));
    index = shr(index, 3);
    let i = index & 7;
    pos[1] = add(add(i, shl(i, 2)), 1);
    index = shr(index, 3);
    let i = index & 7;
    pos[2] = add(add(i, shl(i, 2)), 2);
    index = shr(index, 3);
    let j = index & 1;
    index = shr(index, 1);
    let i = index & 7;
    pos[3] = add(add(add(i, shl(i, 2)), 3), j);

    cod.fill(0);
    for &p in &pos {
        let bit = sign & 1;
        sign = shr(sign, 1);
        cod[p as usize] = if bit != 0 { 8191 } else { -8192 };
    }
}

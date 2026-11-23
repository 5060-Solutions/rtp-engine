//! LP analysis and LSP conversions (LPC.C, LPCFUNC.C).

use super::basic_op::{
    MAX_32, MIN_16, abs_s, add, clear_overflow, div_32, div_s, extract_h, extract_l, l_abs, l_add,
    l_comp, l_extract, l_mac, l_msu, l_mult, l_negate, l_shl, l_shr, l_shr_r, l_sub, mpy_32,
    mpy_32_16, mult, mult_r, negate, norm_l, norm_s, overflow, round, shl, shr, sub,
};
use super::tables::{GRID, HAMWINDOW, LAG_H, LAG_L, SLOPE_ACOS, SLOPE_COS, TABLE2};
use super::{GRID_POINTS, L_WINDOW, M, MP1, NC};

/// Windowed autocorrelation of `x` (`L_WINDOW` samples), as double
/// precision `(r_h, r_l)`.
pub fn autocorr(x: &[i16], r_h: &mut [i16; MP1], r_l: &mut [i16; MP1]) {
    let mut y = [0i16; L_WINDOW];
    for i in 0..L_WINDOW {
        y[i] = mult_r(x[i], HAMWINDOW[i]);
    }
    let mut sum;
    loop {
        clear_overflow();
        sum = 1;
        for &v in &y {
            sum = l_mac(sum, v, v);
        }
        if !overflow() {
            break;
        }
        for v in &mut y {
            *v = shr(*v, 2);
        }
    }
    let norm = norm_l(sum);
    sum = l_shl(sum, norm);
    (r_h[0], r_l[0]) = l_extract(sum);
    for i in 1..=M {
        sum = 0;
        for j in 0..L_WINDOW - i {
            sum = l_mac(sum, y[j], y[j + i]);
        }
        sum = l_shl(sum, norm);
        (r_h[i], r_l[i]) = l_extract(sum);
    }
}

pub fn lag_window(r_h: &mut [i16; MP1], r_l: &mut [i16; MP1]) {
    for i in 1..=M {
        let x = mpy_32(r_h[i], r_l[i], LAG_H[i - 1], LAG_L[i - 1]);
        (r_h[i], r_l[i]) = l_extract(x);
    }
}

/// Levinson-Durbin recursion. Keeps the last stable filter, which it
/// returns in place of an unstable one.
pub struct Levinson {
    old_a: [i16; MP1],
    old_rc: [i16; 2],
}

impl Levinson {
    pub const fn new() -> Self {
        let mut old_a = [0; MP1];
        old_a[0] = 4096;
        Self {
            old_a,
            old_rc: [0; 2],
        }
    }

    pub fn run(&mut self, rh: &[i16; MP1], rl: &[i16; MP1], a: &mut [i16], rc: &mut [i16; M]) {
        let mut ah = [0i16; MP1];
        let mut al = [0i16; MP1];
        let mut anh = [0i16; MP1];
        let mut anl = [0i16; MP1];

        let mut t1 = l_comp(rh[1], rl[1]);
        let mut t2 = l_abs(t1);
        let mut t0 = div_32(t2, rh[0], rl[0]);
        if t1 > 0 {
            t0 = l_negate(t0);
        }
        let (mut kh, mut kl) = l_extract(t0);
        rc[0] = kh;
        t0 = l_shr(t0, 4);
        (ah[1], al[1]) = l_extract(t0);

        t0 = mpy_32(kh, kl, kh, kl);
        t0 = l_abs(t0);
        t0 = l_sub(MAX_32, t0);
        let (hi, lo) = l_extract(t0);
        t0 = mpy_32(rh[0], rl[0], hi, lo);
        let mut alp_exp = norm_l(t0);
        t0 = l_shl(t0, alp_exp);
        let (mut alp_h, mut alp_l) = l_extract(t0);

        for i in 2..=M {
            t0 = 0;
            for j in 1..i {
                t0 = l_add(t0, mpy_32(rh[j], rl[j], ah[i - j], al[i - j]));
            }
            t0 = l_shl(t0, 4);
            t1 = l_comp(rh[i], rl[i]);
            t0 = l_add(t0, t1);

            t1 = l_abs(t0);
            t2 = div_32(t1, alp_h, alp_l);
            if t0 > 0 {
                t2 = l_negate(t2);
            }
            t2 = l_shl(t2, alp_exp);
            (kh, kl) = l_extract(t2);
            rc[i - 1] = kh;

            if sub(abs_s(kh), 32750) > 0 {
                a[..MP1].copy_from_slice(&self.old_a);
                rc[0] = self.old_rc[0];
                rc[1] = self.old_rc[1];
                return;
            }

            for j in 1..i {
                t0 = mpy_32(kh, kl, ah[i - j], al[i - j]);
                t0 = l_add(t0, l_comp(ah[j], al[j]));
                (anh[j], anl[j]) = l_extract(t0);
            }
            t2 = l_shr(t2, 4);
            (anh[i], anl[i]) = l_extract(t2);

            t0 = mpy_32(kh, kl, kh, kl);
            t0 = l_abs(t0);
            t0 = l_sub(MAX_32, t0);
            let (hi, lo) = l_extract(t0);
            t0 = mpy_32(alp_h, alp_l, hi, lo);
            let j = norm_l(t0);
            t0 = l_shl(t0, j);
            (alp_h, alp_l) = l_extract(t0);
            alp_exp = add(alp_exp, j);

            ah[1..=i].copy_from_slice(&anh[1..=i]);
            al[1..=i].copy_from_slice(&anl[1..=i]);
        }

        a[0] = 4096;
        for i in 1..=M {
            t0 = l_comp(ah[i], al[i]);
            a[i] = round(l_shl(t0, 1));
            self.old_a[i] = a[i];
        }
        self.old_rc = [rc[0], rc[1]];
    }
}

/// Chebyshev polynomial evaluation; `wide` selects the reference's
/// `Chebps_10` scaling, used when the coefficients overflowed in Q11.
fn chebps(x: i16, f: &[i16; NC + 1], wide: bool) -> i16 {
    let (b2_init, x_scale, out_shift) = if wide { (128, 256, 7) } else { (256, 512, 6) };
    let mut b2_h: i16 = b2_init;
    let mut b2_l: i16 = 0;
    let mut t0 = l_mult(x, x_scale);
    t0 = l_mac(t0, f[1], 4096);
    let (mut b1_h, mut b1_l) = l_extract(t0);
    for &fi in &f[2..NC] {
        t0 = mpy_32_16(b1_h, b1_l, x);
        t0 = l_shl(t0, 1);
        t0 = l_mac(t0, b2_h, MIN_16);
        t0 = l_msu(t0, b2_l, 1);
        t0 = l_mac(t0, fi, 4096);
        let (b0_h, b0_l) = l_extract(t0);
        b2_l = b1_l;
        b2_h = b1_h;
        b1_l = b0_l;
        b1_h = b0_h;
    }
    t0 = mpy_32_16(b1_h, b1_l, x);
    t0 = l_mac(t0, b2_h, MIN_16);
    t0 = l_msu(t0, b2_l, 1);
    t0 = l_mac(t0, f[NC], 2048);
    t0 = l_shl(t0, out_shift);
    extract_h(t0)
}

/// LP coefficients to line spectral pairs; keeps `old_lsp` if fewer than
/// `M` roots are found.
pub fn az_lsp(a: &[i16], lsp: &mut [i16; M], old_lsp: &[i16; M]) {
    let mut f1 = [0i16; NC + 1];
    let mut f2 = [0i16; NC + 1];
    let mut ovf_coef = false;

    f1[0] = 2048;
    f2[0] = 2048;
    for i in 0..NC {
        clear_overflow();
        let mut t0 = l_mult(a[i + 1], 16384);
        t0 = l_mac(t0, a[M - i], 16384);
        let x = extract_h(t0);
        ovf_coef |= overflow();
        clear_overflow();
        f1[i + 1] = sub(x, f1[i]);
        ovf_coef |= overflow();
        clear_overflow();
        t0 = l_mult(a[i + 1], 16384);
        t0 = l_msu(t0, a[M - i], 16384);
        let x = extract_h(t0);
        ovf_coef |= overflow();
        clear_overflow();
        f2[i + 1] = add(x, f2[i]);
        ovf_coef |= overflow();
    }

    if ovf_coef {
        f1[0] = 1024;
        f2[0] = 1024;
        for i in 0..NC {
            let mut t0 = l_mult(a[i + 1], 8192);
            t0 = l_mac(t0, a[M - i], 8192);
            f1[i + 1] = sub(extract_h(t0), f1[i]);
            t0 = l_mult(a[i + 1], 8192);
            t0 = l_msu(t0, a[M - i], 8192);
            f2[i + 1] = add(extract_h(t0), f2[i]);
        }
    }

    let mut nf = 0;
    let mut use_f2 = false;
    let coef = |f2_sel: bool| if f2_sel { &f2 } else { &f1 };
    let mut xlow = GRID[0];
    let mut ylow = chebps(xlow, coef(use_f2), ovf_coef);
    let mut j = 0;
    while nf < M && j < GRID_POINTS {
        j += 1;
        let mut xhigh = xlow;
        let mut yhigh = ylow;
        xlow = GRID[j];
        ylow = chebps(xlow, coef(use_f2), ovf_coef);

        if l_mult(ylow, yhigh) <= 0 {
            for _ in 0..2 {
                let xmid = add(shr(xlow, 1), shr(xhigh, 1));
                let ymid = chebps(xmid, coef(use_f2), ovf_coef);
                if l_mult(ylow, ymid) <= 0 {
                    yhigh = ymid;
                    xhigh = xmid;
                } else {
                    ylow = ymid;
                    xlow = xmid;
                }
            }

            let x = sub(xhigh, xlow);
            let mut y = sub(yhigh, ylow);
            let xint = if y == 0 {
                xlow
            } else {
                let sign = y;
                y = abs_s(y);
                let exp = norm_s(y);
                y = shl(y, exp);
                y = div_s(16383, y);
                let mut t0 = l_mult(x, y);
                t0 = l_shr(t0, sub(20, exp));
                y = extract_l(t0);
                if sign < 0 {
                    y = negate(y);
                }
                t0 = l_mult(ylow, y);
                t0 = l_shr(t0, 11);
                sub(xlow, extract_l(t0))
            };

            lsp[nf] = xint;
            xlow = xint;
            nf += 1;
            use_f2 = !use_f2;
            ylow = chebps(xlow, coef(use_f2), ovf_coef);
        }
    }

    if nf < M {
        *lsp = *old_lsp;
    }
}

/// Sum/difference polynomial of the LSPs `lsp[start], lsp[start + 2], ...`.
fn get_lsp_pol(lsp: &[i16], start: usize, f: &mut [i32; 6]) {
    f[0] = l_mult(4096, 2048);
    f[1] = l_msu(0, lsp[start], 512);
    let mut fi = 2;
    let mut lp = start + 2;
    for i in 2..=5 {
        f[fi] = f[fi - 2];
        for _ in 1..i {
            let (hi, lo) = l_extract(f[fi - 1]);
            let mut t0 = mpy_32_16(hi, lo, lsp[lp]);
            t0 = l_shl(t0, 1);
            f[fi] = l_add(f[fi], f[fi - 2]);
            f[fi] = l_sub(f[fi], t0);
            fi -= 1;
        }
        f[fi] = l_msu(f[fi], lsp[lp], 512);
        fi += i;
        lp += 2;
    }
}

/// Line spectral pairs to LP coefficients (Q12).
pub fn lsp_az(lsp: &[i16], a: &mut [i16]) {
    let mut f1 = [0i32; 6];
    let mut f2 = [0i32; 6];
    get_lsp_pol(lsp, 0, &mut f1);
    get_lsp_pol(lsp, 1, &mut f2);
    for i in (1..=5).rev() {
        f1[i] = l_add(f1[i], f1[i - 1]);
        f2[i] = l_sub(f2[i], f2[i - 1]);
    }
    a[0] = 4096;
    let mut j = 10;
    for i in 1..=5 {
        a[i] = extract_l(l_shr_r(l_add(f1[i], f2[i]), 13));
        a[j] = extract_l(l_shr_r(l_sub(f1[i], f2[i]), 13));
        j -= 1;
    }
}

/// LSF (Q13, `0..pi`) to LSP (Q15).
pub fn lsf_lsp2(lsf: &[i16], lsp: &mut [i16]) {
    for (f, p) in lsf.iter().zip(lsp.iter_mut()).take(M) {
        let freq = mult(*f, 20861);
        let ind = shr(freq, 8).clamp(0, 63) as usize;
        let offset = freq & 0x00ff;
        let t = l_mult(SLOPE_COS[ind], offset);
        *p = add(TABLE2[ind], extract_l(l_shr(t, 13)));
    }
}

/// LSP (Q15) to LSF (Q13, `0..pi`).
pub fn lsp_lsf2(lsp: &[i16], lsf: &mut [i16]) {
    let mut ind: i16 = 63;
    for i in (0..M).rev() {
        while sub(TABLE2[ind as usize], lsp[i]) < 0 {
            ind = sub(ind, 1);
            if ind <= 0 {
                break;
            }
        }
        let offset = sub(lsp[i], TABLE2[ind as usize]);
        let t = l_mult(SLOPE_ACOS[ind as usize], offset);
        let freq = add(shl(ind, 9), extract_l(l_shr(t, 12)));
        lsf[i] = mult(freq, 25736);
    }
}

/// Bandwidth expansion: `ap[i] = a[i] * gamma^i`.
pub fn weight_az(a: &[i16], gamma: i16, ap: &mut [i16]) {
    ap[0] = a[0];
    let mut fac = gamma;
    for i in 1..M {
        ap[i] = round(l_mult(a[i], fac));
        fac = round(l_mult(fac, gamma));
    }
    ap[M] = round(l_mult(a[M], fac));
}

/// LP filters for both subframes: the first from the midpoint of the old
/// and new LSPs, the second from the new LSPs.
pub fn int_qlpc(lsp_old: &[i16; M], lsp_new: &[i16; M], az: &mut [i16; 2 * MP1]) {
    let mut lsp = [0i16; M];
    for i in 0..M {
        lsp[i] = add(shr(lsp_new[i], 1), shr(lsp_old[i], 1));
    }
    lsp_az(&lsp, &mut az[..MP1]);
    lsp_az(lsp_new, &mut az[MP1..]);
}

//! Math helpers, filters and the pre/post-processing high-pass filters
//! (DSPFUNC.C, FILTER.C, UTIL.C, PRE_PROC.C, POST_PRO.C).

use super::M;
use super::basic_op::{
    add, extract_h, extract_l, l_add, l_deposit_h, l_extract, l_mac, l_msu, l_mult, l_shl, l_shr,
    l_shr_r, mpy_32_16, norm_l, round, shr, sub,
};
use super::tables::{A100, A140, B100, B140, TABLOG, TABPOW, TABSQR};

/// `2^(exponent.fraction)`, `fraction` in Q15.
pub fn pow2(exponent: i16, fraction: i16) -> i32 {
    let mut x = l_mult(fraction, 32);
    let i = extract_h(x) as usize;
    x = l_shr(x, 1);
    let a = extract_l(x) & 0x7fff;
    x = l_deposit_h(TABPOW[i]);
    let tmp = sub(TABPOW[i], TABPOW[i + 1]);
    x = l_msu(x, tmp, a);
    l_shr_r(x, sub(30, exponent))
}

/// `log2(x)` as (integer part, Q15 fraction).
pub fn log2(x: i32) -> (i16, i16) {
    if x <= 0 {
        return (0, 0);
    }
    let exp = norm_l(x);
    let mut x = l_shl(x, exp);
    let exponent = sub(30, exp);
    x = l_shr(x, 9);
    let i = sub(extract_h(x), 32) as usize;
    x = l_shr(x, 1);
    let a = extract_l(x) & 0x7fff;
    let mut y = l_deposit_h(TABLOG[i]);
    let tmp = sub(TABLOG[i], TABLOG[i + 1]);
    y = l_msu(y, tmp, a);
    (exponent, extract_h(y))
}

/// `1/sqrt(x)` in Q30.
pub fn inv_sqrt(x: i32) -> i32 {
    if x <= 0 {
        return 0x3fff_ffff;
    }
    let mut exp = norm_l(x);
    let mut x = l_shl(x, exp);
    exp = sub(30, exp);
    if exp & 1 == 0 {
        x = l_shr(x, 1);
    }
    exp = add(shr(exp, 1), 1);
    x = l_shr(x, 9);
    let i = sub(extract_h(x), 16) as usize;
    x = l_shr(x, 1);
    let a = extract_l(x) & 0x7fff;
    let mut y = l_deposit_h(TABSQR[i]);
    let tmp = sub(TABSQR[i], TABSQR[i + 1]);
    y = l_msu(y, tmp, a);
    l_shr(y, exp)
}

/// The reference's pseudo-random generator, used for frame erasure.
pub struct Random(i16);

impl Random {
    pub const fn new() -> Self {
        Self(21845)
    }

    pub const fn with_seed(seed: i16) -> Self {
        Self(seed)
    }

    pub fn draw(&mut self) -> i16 {
        self.0 = extract_l(l_add(l_shr(l_mult(self.0, 31821), 1), 13849));
        self.0
    }
}

/// Synthesis filter `1/A(z)` over `x`, into `y`, starting from `mem`.
///
/// `x` and `y` must have the same length (at most 100). With `update`,
/// `mem` takes the last `M` outputs.
pub fn syn_filt(a: &[i16], x: &[i16], y: &mut [i16], mem: &mut [i16], update: bool) {
    let lg = x.len();
    let mut tmp = [0i16; 100 + M];
    tmp[..M].copy_from_slice(&mem[..M]);
    for i in 0..lg {
        let mut s = l_mult(x[i], a[0]);
        for j in 1..=M {
            s = l_msu(s, a[j], tmp[M + i - j]);
        }
        s = l_shl(s, 3);
        tmp[M + i] = round(s);
    }
    y[..lg].copy_from_slice(&tmp[M..M + lg]);
    if update {
        mem[..M].copy_from_slice(&y[lg - M..lg]);
    }
}

/// LP residual of `x[pos..pos + y.len()]` into `y`; `x[pos - M..pos]` is history.
pub fn residu(a: &[i16], x: &[i16], pos: usize, y: &mut [i16]) {
    for (n, out) in (pos..).zip(y.iter_mut()) {
        let mut acc = l_mult(x[n], a[0]);
        for j in 1..=M {
            acc = l_mac(acc, a[j], x[n - j]);
        }
        acc = l_shl(acc, 3);
        *out = round(acc);
    }
}

/// Second-order IIR high-pass shared by the pre- and post-processing.
struct HighPass {
    y2_hi: i16,
    y2_lo: i16,
    y1_hi: i16,
    y1_lo: i16,
    x0: i16,
    x1: i16,
}

impl HighPass {
    const fn new() -> Self {
        Self {
            y2_hi: 0,
            y2_lo: 0,
            y1_hi: 0,
            y1_lo: 0,
            x0: 0,
            x1: 0,
        }
    }

    /// Filter in place. `b`/`a` are the coefficients; the reference scales
    /// the pre-filter's accumulator by `<< 3` and the post-filter's by
    /// `<< 2`, rounding the output after one more `<< 1`.
    fn run(&mut self, signal: &mut [i16], b: [i16; 3], a: [i16; 3], post: bool) {
        for s in signal.iter_mut() {
            let x2 = self.x1;
            self.x1 = self.x0;
            self.x0 = *s;
            let mut t = mpy_32_16(self.y1_hi, self.y1_lo, a[1]);
            t = l_add(t, mpy_32_16(self.y2_hi, self.y2_lo, a[2]));
            t = l_mac(t, self.x0, b[0]);
            t = l_mac(t, self.x1, b[1]);
            t = l_mac(t, x2, b[2]);
            if post {
                t = l_shl(t, 2);
                *s = round(l_shl(t, 1));
            } else {
                t = l_shl(t, 3);
                *s = round(t);
            }
            self.y2_hi = self.y1_hi;
            self.y2_lo = self.y1_lo;
            (self.y1_hi, self.y1_lo) = l_extract(t);
        }
    }
}

/// Encoder input filter: 140 Hz high-pass and divide by two.
pub struct PreProcess(HighPass);

impl PreProcess {
    pub const fn new() -> Self {
        Self(HighPass::new())
    }

    pub fn run(&mut self, signal: &mut [i16]) {
        self.0.run(signal, B140, A140, false);
    }
}

/// Decoder output filter: 100 Hz high-pass and multiply by two.
pub struct PostProcess(HighPass);

impl PostProcess {
    pub const fn new() -> Self {
        Self(HighPass::new())
    }

    pub fn run(&mut self, signal: &mut [i16]) {
        self.0.run(signal, B100, A100, true);
    }
}

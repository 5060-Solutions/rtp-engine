//! ITU-T fixed-point basic operators (BASIC_OP.C, OPER_32B.C).
//!
//! G.729 is specified by its fixed-point reference code, so bit-exactness
//! depends on reproducing these operators exactly, saturation included. The
//! reference keeps a global `Overflow` flag that a few routines clear, run a
//! computation, and then test; here it is a thread-local, which is safe
//! because every such clear-compute-test sequence runs synchronously inside
//! one codec call on one thread.

use std::cell::Cell;

pub const MAX_16: i16 = i16::MAX;
pub const MIN_16: i16 = i16::MIN;
pub const MAX_32: i32 = i32::MAX;
pub const MIN_32: i32 = i32::MIN;

thread_local! {
    static OVERFLOW: Cell<bool> = const { Cell::new(false) };
}

#[inline]
fn set_overflow() {
    OVERFLOW.with(|o| o.set(true));
}

/// Clear the overflow flag.
#[inline]
pub fn clear_overflow() {
    OVERFLOW.with(|o| o.set(false));
}

/// Whether an operator saturated since the flag was last cleared.
#[inline]
pub fn overflow() -> bool {
    OVERFLOW.with(Cell::get)
}

#[inline]
fn sature(x: i32) -> i16 {
    if x > 0x7fff {
        set_overflow();
        MAX_16
    } else if x < -0x8000 {
        set_overflow();
        MIN_16
    } else {
        x as i16
    }
}

#[inline]
pub fn add(a: i16, b: i16) -> i16 {
    sature(i32::from(a) + i32::from(b))
}

#[inline]
pub fn sub(a: i16, b: i16) -> i16 {
    sature(i32::from(a) - i32::from(b))
}

#[inline]
pub fn abs_s(a: i16) -> i16 {
    if a == MIN_16 { MAX_16 } else { a.abs() }
}

#[inline]
pub fn negate(a: i16) -> i16 {
    if a == MIN_16 { MAX_16 } else { -a }
}

#[inline]
pub fn shl(a: i16, n: i16) -> i16 {
    if n < 0 {
        return shr(a, -n);
    }
    let r = i64::from(a) << n.min(40);
    if (n > 15 && a != 0) || r != i64::from(r as i16) {
        set_overflow();
        if a > 0 { MAX_16 } else { MIN_16 }
    } else {
        r as i16
    }
}

#[inline]
pub fn shr(a: i16, n: i16) -> i16 {
    if n < 0 {
        shl(a, -n)
    } else if n >= 15 {
        if a < 0 { -1 } else { 0 }
    } else {
        a >> n
    }
}

#[inline]
pub fn mult(a: i16, b: i16) -> i16 {
    let p = (i32::from(a) * i32::from(b)) >> 15;
    sature(p)
}

#[inline]
pub fn mult_r(a: i16, b: i16) -> i16 {
    let p = (i32::from(a) * i32::from(b) + 0x4000) >> 15;
    sature(p)
}

#[inline]
pub fn l_mult(a: i16, b: i16) -> i32 {
    let p = i32::from(a) * i32::from(b);
    if p == 0x4000_0000 {
        set_overflow();
        MAX_32
    } else {
        p * 2
    }
}

#[inline]
pub fn extract_h(x: i32) -> i16 {
    (x >> 16) as i16
}

#[inline]
pub fn extract_l(x: i32) -> i16 {
    x as i16
}

#[inline]
pub fn round(x: i32) -> i16 {
    extract_h(l_add(x, 0x8000))
}

#[inline]
pub fn l_add(a: i32, b: i32) -> i32 {
    a.checked_add(b).unwrap_or_else(|| {
        set_overflow();
        if a < 0 { MIN_32 } else { MAX_32 }
    })
}

#[inline]
pub fn l_sub(a: i32, b: i32) -> i32 {
    a.checked_sub(b).unwrap_or_else(|| {
        set_overflow();
        if a < 0 { MIN_32 } else { MAX_32 }
    })
}

#[inline]
pub fn l_mac(acc: i32, a: i16, b: i16) -> i32 {
    l_add(acc, l_mult(a, b))
}

#[inline]
pub fn l_msu(acc: i32, a: i16, b: i16) -> i32 {
    l_sub(acc, l_mult(a, b))
}

#[inline]
pub fn l_negate(x: i32) -> i32 {
    if x == MIN_32 { MAX_32 } else { -x }
}

#[inline]
pub fn l_abs(x: i32) -> i32 {
    if x == MIN_32 { MAX_32 } else { x.abs() }
}

pub fn l_shl(mut x: i32, n: i16) -> i32 {
    if n <= 0 {
        return l_shr(x, -n);
    }
    for _ in 0..n {
        if x > 0x3fff_ffff {
            set_overflow();
            return MAX_32;
        }
        if x < -0x4000_0000 {
            set_overflow();
            return MIN_32;
        }
        x *= 2;
    }
    x
}

#[inline]
pub fn l_shr(x: i32, n: i16) -> i32 {
    if n < 0 {
        l_shl(x, -n)
    } else if n >= 31 {
        if x < 0 { -1 } else { 0 }
    } else {
        x >> n
    }
}

#[inline]
pub fn l_shr_r(x: i32, n: i16) -> i32 {
    if n > 31 {
        return 0;
    }
    let mut out = l_shr(x, n);
    if n > 0 && (x & (1i32 << (n - 1))) != 0 {
        out = out.wrapping_add(1);
    }
    out
}

#[inline]
pub fn l_deposit_h(a: i16) -> i32 {
    i32::from(a) << 16
}

#[inline]
pub fn l_deposit_l(a: i16) -> i32 {
    i32::from(a)
}

#[inline]
pub fn norm_s(a: i16) -> i16 {
    match a {
        0 => 0,
        -1 => 15,
        _ => {
            let v = if a < 0 { !a } else { a };
            (v.leading_zeros() - 1) as i16
        }
    }
}

#[inline]
pub fn norm_l(x: i32) -> i16 {
    match x {
        0 => 0,
        -1 => 31,
        _ => {
            let v = if x < 0 { !x } else { x };
            (v.leading_zeros() - 1) as i16
        }
    }
}

/// Fractional division `num / den` in Q15, for `0 <= num <= den`.
///
/// The reference aborts the program on other inputs; a decoder must not
/// panic on what a peer sends, so this saturates instead. No conformant
/// computation reaches that case.
pub fn div_s(num: i16, den: i16) -> i16 {
    if num <= 0 || den <= 0 {
        return 0;
    }
    if num >= den {
        return MAX_16;
    }
    let mut out: i16 = 0;
    let mut n = i32::from(num);
    let d = i32::from(den);
    for _ in 0..15 {
        out <<= 1;
        n <<= 1;
        if n >= d {
            n -= d;
            out += 1;
        }
    }
    out
}

/// Split a 32-bit value into the reference's double-precision `hi`/`lo`.
#[inline]
pub fn l_extract(x: i32) -> (i16, i16) {
    let hi = extract_h(x);
    let lo = extract_l(l_msu(l_shr(x, 1), hi, 16384));
    (hi, lo)
}

#[inline]
pub fn l_comp(hi: i16, lo: i16) -> i32 {
    l_mac(l_deposit_h(hi), lo, 1)
}

#[inline]
pub fn mpy_32(hi1: i16, lo1: i16, hi2: i16, lo2: i16) -> i32 {
    let mut x = l_mult(hi1, hi2);
    x = l_mac(x, mult(hi1, lo2), 1);
    l_mac(x, mult(lo1, hi2), 1)
}

#[inline]
pub fn mpy_32_16(hi: i16, lo: i16, n: i16) -> i32 {
    l_mac(l_mult(hi, n), mult(lo, n), 1)
}

pub fn div_32(num: i32, den_hi: i16, den_lo: i16) -> i32 {
    let approx = div_s(0x3fff, den_hi);
    let x = l_sub(MAX_32, mpy_32_16(den_hi, den_lo, approx));
    let (hi, lo) = l_extract(x);
    let (hi, lo) = l_extract(mpy_32_16(hi, lo, approx));
    let (n_hi, n_lo) = l_extract(num);
    l_shl(mpy_32(n_hi, n_lo, hi, lo), 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturating_ops_set_overflow() {
        clear_overflow();
        assert_eq!(add(MAX_16, 1), MAX_16);
        assert!(overflow());
        clear_overflow();
        assert_eq!(l_mult(MIN_16, MIN_16), MAX_32);
        assert!(overflow());
        clear_overflow();
        assert_eq!(l_shl(0x4000_0000, 1), MAX_32);
        assert!(overflow());
        clear_overflow();
        assert_eq!(sub(1, 2), -1);
        assert!(!overflow());
    }

    #[test]
    fn shifts_match_reference() {
        assert_eq!(shr(-1, 3), -1);
        assert_eq!(shr(-9, 1), -5);
        assert_eq!(shl(0x4000, 1), MAX_16);
        assert_eq!(shl(-0x4000, 1), MIN_16);
        assert_eq!(shl(3, -1), 1);
        assert_eq!(l_shr_r(3, 1), 2);
        assert_eq!(l_shr(-1, 40), -1);
    }

    #[test]
    fn norms_and_division() {
        assert_eq!(norm_s(1), 14);
        assert_eq!(norm_s(-2), 14);
        assert_eq!(norm_s(-1), 15);
        assert_eq!(norm_l(1), 30);
        assert_eq!(norm_l(MIN_32), 0);
        assert_eq!(div_s(1, 2), 0x4000);
        assert_eq!(div_s(5, 5), MAX_16);
        assert_eq!(div_s(0, 5), 0);
    }

    #[test]
    fn mult_rounds_like_reference() {
        assert_eq!(mult(-1, 1), -1);
        assert_eq!(mult_r(-1, 1), 0);
        assert_eq!(mult(MIN_16, MIN_16), MAX_16);
        assert_eq!(round(0x7fff_ffff), MAX_16);
    }

    #[test]
    fn double_precision_roundtrip() {
        for x in [0, 1, -1, 0x1234_5678, -0x1234_5678, MAX_32 >> 1] {
            let (hi, lo) = l_extract(x);
            assert_eq!(l_comp(hi, lo), x & !1);
        }
    }
}

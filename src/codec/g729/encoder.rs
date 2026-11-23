//! The G.729 Annex A encoder (CODER.C, COD_LD8A.C).

use super::acelp::acelp_code_a;
use super::basic_op::{add, extract_h, l_mac, l_mult, l_shl, mult, negate, round, sub};
use super::dsp::{PreProcess, residu, syn_filt};
use super::gain::{GainQuantizer, corr_xy2};
use super::lpc::{Levinson, autocorr, az_lsp, int_qlpc, lag_window, weight_az};
use super::lsp::LspEncoder;
use super::pitch::{
    EXC_HISTORY, Taming, enc_lag3, g_pitch, parity_pitch, pitch_fr3_fast, pitch_ol_fast,
};
use super::{
    GAMMA1, L_FRAME, L_NEXT, L_SUBFR, L_TOTAL, LSP_INIT, M, MP1, PIT_MAX, PIT_MIN, PRM_SIZE,
    SHARPMAX, SHARPMIN,
};

const GPCLIP: i16 = 15564;

/// Where the current frame starts in the speech buffer, and the 5 ms of
/// look-ahead the LP analysis window covers before it.
const NEW_SPEECH: usize = L_TOTAL - L_FRAME;
const SPEECH: usize = NEW_SPEECH - L_NEXT;
/// Start of the current frame in the excitation buffer.
const EXC: usize = EXC_HISTORY;

pub struct Encoder {
    pre_process: PreProcess,
    old_speech: [i16; L_TOTAL],
    old_wsp: [i16; L_FRAME + PIT_MAX],
    old_exc: [i16; L_FRAME + EXC_HISTORY],
    lsp_old: [i16; M],
    lsp_old_q: [i16; M],
    mem_w0: [i16; M],
    mem_w: [i16; M],
    sharp: i16,
    levinson: Levinson,
    lsp: LspEncoder,
    taming: Taming,
    gain: GainQuantizer,
}

impl Encoder {
    pub const fn new() -> Self {
        Self {
            pre_process: PreProcess::new(),
            old_speech: [0; L_TOTAL],
            old_wsp: [0; L_FRAME + PIT_MAX],
            old_exc: [0; L_FRAME + EXC_HISTORY],
            lsp_old: LSP_INIT,
            lsp_old_q: LSP_INIT,
            mem_w0: [0; M],
            mem_w: [0; M],
            sharp: SHARPMIN,
            levinson: Levinson::new(),
            lsp: LspEncoder::new(),
            taming: Taming::new(),
            gain: GainQuantizer::new(),
        }
    }

    /// Encode one 10 ms frame of 8 kHz PCM into its 11 parameters.
    pub fn encode_frame(&mut self, pcm: &[i16]) -> [i16; PRM_SIZE] {
        let mut ana = [0i16; PRM_SIZE];
        let new = &mut self.old_speech[NEW_SPEECH..];
        new.copy_from_slice(&pcm[..L_FRAME]);
        self.pre_process.run(new);

        // LP analysis, quantisation and interpolation.
        let mut aq_t = [0i16; 2 * MP1];
        let mut ap_t = [0i16; 2 * MP1];
        {
            let mut r_h = [0i16; MP1];
            let mut r_l = [0i16; MP1];
            let mut rc = [0i16; M];
            let mut lsp_new = [0i16; M];
            let mut lsp_new_q = [0i16; M];
            autocorr(&self.old_speech, &mut r_h, &mut r_l);
            lag_window(&mut r_h, &mut r_l);
            self.levinson.run(&r_h, &r_l, &mut ap_t, &mut rc);
            az_lsp(&ap_t, &mut lsp_new, &self.lsp_old);
            self.lsp.quantize(&lsp_new, &mut lsp_new_q, &mut ana[..2]);
            int_qlpc(&self.lsp_old_q, &lsp_new_q, &mut aq_t);
            weight_az(&aq_t[..MP1], GAMMA1, &mut ap_t[..MP1]);
            weight_az(&aq_t[MP1..], GAMMA1, &mut ap_t[MP1..]);
            self.lsp_old = lsp_new;
            self.lsp_old_q = lsp_new_q;
        }

        // Residual, then the perceptually weighted speech.
        for sf in 0..2 {
            let o = sf * L_SUBFR;
            residu(
                &aq_t[sf * MP1..],
                &self.old_speech,
                SPEECH + o,
                &mut self.old_exc[EXC + o..EXC + o + L_SUBFR],
            );
        }
        for sf in 0..2 {
            let o = sf * L_SUBFR;
            let ap = &ap_t[sf * MP1..(sf + 1) * MP1];
            let mut ap1 = [0i16; MP1];
            ap1[0] = 4096;
            for i in 1..=M {
                ap1[i] = sub(ap[i], mult(ap[i - 1], 22938));
            }
            syn_filt(
                &ap1,
                &self.old_exc[EXC + o..EXC + o + L_SUBFR],
                &mut self.old_wsp[PIT_MAX + o..PIT_MAX + o + L_SUBFR],
                &mut self.mem_w,
                true,
            );
        }

        // Open-loop pitch and the closed-loop search range.
        let t_op = pitch_ol_fast(&self.old_wsp);
        let mut t0_min = sub(t_op, 3).max(PIT_MIN as i16);
        let mut t0_max = add(t0_min, 6);
        if sub(t0_max, PIT_MAX as i16) > 0 {
            t0_max = PIT_MAX as i16;
            t0_min = sub(t0_max, 6);
        }

        let mut k = 2;
        for sf in 0..2 {
            let i_subfr = sf * L_SUBFR;
            let ap = &ap_t[sf * MP1..(sf + 1) * MP1];
            let pos = EXC + i_subfr;

            // Impulse response of the weighted synthesis filter.
            let mut h_in = [0i16; L_SUBFR];
            h_in[0] = 4096;
            let mut h1 = [0i16; L_SUBFR];
            syn_filt(ap, &h_in, &mut h1, &mut [0; M], false);

            // Target for the adaptive codebook.
            let mut xn = [0i16; L_SUBFR];
            let mut mem = self.mem_w0;
            syn_filt(
                ap,
                &self.old_exc[pos..pos + L_SUBFR],
                &mut xn,
                &mut mem,
                false,
            );

            let mut t0_frac = 0;
            let t0 = pitch_fr3_fast(
                &mut self.old_exc,
                pos,
                &xn,
                &h1,
                t0_min,
                t0_max,
                i_subfr,
                &mut t0_frac,
            );
            let index = enc_lag3(t0, t0_frac, &mut t0_min, &mut t0_max, sf == 0);
            ana[k] = index;
            k += 1;
            if sf == 0 {
                ana[k] = parity_pitch(index);
                k += 1;
            }

            // Filtered adaptive codebook vector and its gain.
            let mut y1 = [0i16; L_SUBFR];
            syn_filt(
                ap,
                &self.old_exc[pos..pos + L_SUBFR],
                &mut y1,
                &mut [0; M],
                false,
            );
            let mut g_coeff = [0i16; 4];
            let mut gain_pit = g_pitch(&xn, &y1, &mut g_coeff);

            let taming = self.taming.test_err(t0, t0_frac);
            if taming {
                gain_pit = gain_pit.min(GPCLIP);
            }

            // Target for the fixed codebook.
            let mut xn2 = [0i16; L_SUBFR];
            for i in 0..L_SUBFR {
                let t = l_shl(l_mult(y1[i], gain_pit), 1);
                xn2[i] = sub(xn[i], extract_h(t));
            }

            let mut code = [0i16; L_SUBFR];
            let mut y2 = [0i16; L_SUBFR];
            let mut sign = 0;
            let index = acelp_code_a(&xn2, &mut h1, t0, self.sharp, &mut code, &mut y2, &mut sign);
            ana[k] = index;
            ana[k + 1] = sign;
            k += 2;

            // Joint gain quantisation.
            let mut g_coeff_cs = [0i16; 5];
            let mut exp_g_coeff_cs = [0i16; 5];
            g_coeff_cs[0] = g_coeff[0];
            exp_g_coeff_cs[0] = negate(g_coeff[1]);
            g_coeff_cs[1] = negate(g_coeff[2]);
            exp_g_coeff_cs[1] = negate(add(g_coeff[3], 1));
            corr_xy2(&xn, &y1, &y2, &mut g_coeff_cs, &mut exp_g_coeff_cs);
            let mut gain_code = 0;
            ana[k] = self.gain.quantize(
                &code,
                &g_coeff_cs,
                &exp_g_coeff_cs,
                &mut gain_pit,
                &mut gain_code,
                taming,
            );
            k += 1;

            self.sharp = gain_pit.clamp(SHARPMIN, SHARPMAX);

            // Excitation for the next subframe and the filter memories.
            for i in 0..L_SUBFR {
                let mut t = l_mult(self.old_exc[pos + i], gain_pit);
                t = l_mac(t, code[i], gain_code);
                t = l_shl(t, 1);
                self.old_exc[pos + i] = round(t);
            }
            self.taming.update(gain_pit, t0);

            for (j, i) in (L_SUBFR - M..L_SUBFR).enumerate() {
                let temp = extract_h(l_shl(l_mult(y1[i], gain_pit), 1));
                let k = extract_h(l_shl(l_mult(y2[i], gain_code), 2));
                self.mem_w0[j] = sub(xn[i], add(temp, k));
            }
        }

        self.old_speech.copy_within(L_FRAME.., 0);
        self.old_wsp.copy_within(L_FRAME.., 0);
        self.old_exc.copy_within(L_FRAME.., 0);
        ana
    }
}

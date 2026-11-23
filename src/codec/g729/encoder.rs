//! The G.729 Annex A encoder, with Annex B's VAD and DTX (CODER.C,
//! COD_LD8A.C).

use super::acelp::acelp_code_a;
use super::basic_op::{add, extract_h, l_mac, l_mult, l_shl, mult, negate, round, sub};
use super::dsp::{PreProcess, Random, residu, syn_filt};
use super::dtx::{CngContext, CngEncoder, INIT_SEED};
use super::gain::{GainQuantizer, corr_xy2};
use super::lpc::{Levinson, NP, autocorr, az_lsp, int_qlpc, lag_window, lsp_lsf, weight_az};
use super::lsp::LspEncoder;
use super::pitch::{
    EXC_HISTORY, Taming, enc_lag3, g_pitch, parity_pitch, pitch_fr3_fast, pitch_ol_fast,
};
use super::vad::{Vad, VadInput};
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

/// One encoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoded {
    /// Active speech: the 11 parameters.
    Speech([i16; PRM_SIZE]),
    /// Annex B silence insertion descriptor: the 4 parameters.
    Sid([i16; 4]),
    /// Annex B: nothing to send for this frame.
    NotSent,
}

/// The VAD's decisions for the previous two frames, which it feeds back
/// into its next decision.
struct VadHistory {
    prev: bool,
    prev2: bool,
}

impl VadHistory {
    fn push(&mut self, vad: bool) {
        self.prev2 = self.prev;
        self.prev = vad;
    }
}

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
    /// Annex B: send SID frames and nothing during silence.
    dtx: bool,
    vad: Vad,
    cng: CngEncoder,
    vad_history: VadHistory,
    /// Whether the last frame went out as speech. With DTX on from the
    /// start this is the previous VAD decision; it differs only when DTX is switched on
    /// mid-call, where the first silent frame must still open with a SID.
    sent_speech: bool,
    seed: Random,
    frame: i16,
}

impl Encoder {
    pub fn new(dtx: bool) -> Self {
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
            dtx,
            vad: Vad::new(),
            cng: CngEncoder::new(),
            vad_history: VadHistory {
                prev: true,
                prev2: true,
            },
            sent_speech: true,
            seed: Random::with_seed(INIT_SEED),
            frame: 0,
        }
    }

    /// Turn DTX on or off from the next frame. The VAD and comfort-noise
    /// state run either way, so switching needs no resynchronisation.
    pub fn set_dtx(&mut self, dtx: bool) {
        self.dtx = dtx;
    }

    /// Encode one 10 ms frame of 8 kHz PCM.
    pub fn encode_frame(&mut self, pcm: &[i16]) -> Encoded {
        let mut ana = [0i16; PRM_SIZE];
        let new = &mut self.old_speech[NEW_SPEECH..];
        new.copy_from_slice(&pcm[..L_FRAME]);
        self.pre_process.run(new);
        self.frame = if self.frame == i16::MAX {
            256
        } else {
            self.frame + 1
        };

        // LP analysis, quantisation and interpolation.
        let mut aq_t = [0i16; 2 * MP1];
        let mut ap_t = [0i16; 2 * MP1];
        {
            let mut r_h = [0i16; NP + 1];
            let mut r_l = [0i16; NP + 1];
            let mut rc = [0i16; M];
            let mut lsp_new = [0i16; M];
            let mut lsp_new_q = [0i16; M];
            let exp_r0 = autocorr(&self.old_speech, &mut r_h, &mut r_l);
            let rh_nbe = r_h;
            lag_window(&mut r_h, &mut r_l);
            self.levinson.run(&r_h, &r_l, &mut ap_t, &mut rc, &mut 0);
            az_lsp(&ap_t, &mut lsp_new, &self.lsp_old);

            // Annex B: voice activity, run whether or not DTX is on so its
            // state always tracks the signal.
            let mut lsf_new = [0i16; M];
            lsp_lsf(&lsp_new, &mut lsf_new);
            let vad = self.vad.detect(&VadInput {
                rc: rc[1],
                lsf: &lsf_new,
                r_h: &r_h,
                r_l: &r_l,
                exp_r0,
                sigpp: &self.old_speech,
                frm_count: self.frame,
                prev_marker: self.vad_history.prev,
                pprev_marker: self.vad_history.prev2,
            });
            self.cng.update(&rh_nbe, exp_r0, vad);

            if !vad && self.dtx {
                return self.encode_silence();
            }
            self.seed = Random::with_seed(INIT_SEED);
            self.vad_history.push(vad);
            self.sent_speech = true;

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

        self.shift_buffers();
        Encoded::Speech(ana)
    }

    /// Annex B: an inactive frame. Generates the comfort noise the decoder
    /// will, so the filter memories stay in step with it.
    fn encode_silence(&mut self) -> Encoded {
        let mut aq_t = [0i16; 2 * MP1];
        let frame = self.cng.encode(CngContext {
            exc: &mut self.old_exc,
            pos: EXC,
            past_vad: self.sent_speech,
            lsp_old_q: &mut self.lsp_old_q,
            aq: &mut aq_t,
            freq_prev: self.lsp.freq_prev_mut(),
            seed: &mut self.seed,
            levinson: &mut self.levinson,
            taming: &mut self.taming,
        });
        self.vad_history.push(false);
        self.sent_speech = false;

        for sf in 0..2 {
            let o = sf * L_SUBFR;
            let aq = &aq_t[sf * MP1..(sf + 1) * MP1];
            let mut xn = [0i16; L_SUBFR];
            residu(aq, &self.old_speech, SPEECH + o, &mut xn);
            let mut ap = [0i16; MP1];
            weight_az(aq, GAMMA1, &mut ap);
            let mut ap1 = [0i16; MP1];
            ap1[0] = 4096;
            for i in 1..=M {
                ap1[i] = sub(ap[i], mult(ap[i - 1], 22938));
            }
            syn_filt(
                &ap1,
                &xn,
                &mut self.old_wsp[PIT_MAX + o..PIT_MAX + o + L_SUBFR],
                &mut self.mem_w,
                true,
            );
            for i in 0..L_SUBFR {
                xn[i] = sub(xn[i], self.old_exc[EXC + o + i]);
            }
            let x = xn;
            syn_filt(&ap, &x, &mut xn, &mut self.mem_w0, true);
        }
        self.sharp = SHARPMIN;
        self.shift_buffers();

        frame.map_or(Encoded::NotSent, Encoded::Sid)
    }

    fn shift_buffers(&mut self) {
        self.old_speech.copy_within(L_FRAME.., 0);
        self.old_wsp.copy_within(L_FRAME.., 0);
        self.old_exc.copy_within(L_FRAME.., 0);
    }
}

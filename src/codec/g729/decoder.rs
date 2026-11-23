//! The G.729 Annex A decoder (DECODER.C, DEC_LD8A.C), postfilter and
//! output filter included.

use super::acelp::decod_acelp;
use super::basic_op::{add, clear_overflow, l_mac, l_mult, l_shl, mult, overflow, round, shl, shr};
use super::dsp::{PostProcess, Random, syn_filt};
use super::gain::GainDecoder;
use super::lpc::int_qlpc;
use super::lsp::LspDecoder;
use super::pitch::{EXC_HISTORY, check_parity_pitch, dec_lag3, pred_lt_3};
use super::postfilter::PostFilter;
use super::{L_FRAME, L_SUBFR, LSP_INIT, M, MP1, PIT_MAX, PRM_SIZE, SHARPMAX, SHARPMIN};

/// Start of the current frame in the excitation buffer.
const EXC: usize = EXC_HISTORY;

pub struct Decoder {
    old_exc: [i16; L_FRAME + EXC_HISTORY],
    lsp_old: [i16; M],
    mem_syn: [i16; M],
    sharp: i16,
    old_t0: i16,
    gain_code: i16,
    gain_pitch: i16,
    lsp: LspDecoder,
    gain: GainDecoder,
    random: Random,
    post_filter: PostFilter,
    post_process: PostProcess,
}

impl Decoder {
    pub const fn new() -> Self {
        Self {
            old_exc: [0; L_FRAME + EXC_HISTORY],
            lsp_old: LSP_INIT,
            mem_syn: [0; M],
            sharp: SHARPMIN,
            old_t0: 60,
            gain_code: 0,
            gain_pitch: 0,
            lsp: LspDecoder::new(),
            gain: GainDecoder::new(),
            random: Random::new(),
            post_filter: PostFilter::new(),
            post_process: PostProcess::new(),
        }
    }

    /// Decode one frame's parameters to 80 samples. With `bfi` (bad frame
    /// indicator) the parameters are ignored and the frame is concealed.
    pub fn decode_frame(&mut self, prm: &[i16; PRM_SIZE], bfi: bool) -> [i16; L_FRAME] {
        let mut prm = *prm;
        prm[3] = check_parity_pitch(prm[2], prm[3]);

        let mut lsp_new = [0i16; M];
        self.lsp.decode(&prm[..2], &mut lsp_new, bfi);
        let mut a_t = [0i16; 2 * MP1];
        int_qlpc(&self.lsp_old, &lsp_new, &mut a_t);
        self.lsp_old = lsp_new;

        let mut synth = [0i16; L_FRAME];
        let mut t2 = [0i16; 2];
        let mut t0 = 0;
        let mut t0_frac = 0;
        let mut p = 2;

        for sf in 0..2 {
            let i_subfr = sf * L_SUBFR;
            let pos = EXC + i_subfr;
            let az = &a_t[sf * MP1..(sf + 1) * MP1];

            // Pitch lag; repeated (and slowly lengthened) on a bad frame or
            // a failed parity check.
            let index = prm[p];
            p += 1;
            let bad_pitch = if sf == 0 {
                let parity_err = prm[p];
                p += 1;
                bfi || parity_err != 0
            } else {
                bfi
            };
            if bad_pitch {
                t0 = self.old_t0;
                t0_frac = 0;
                self.old_t0 = add(self.old_t0, 1).min(PIT_MAX as i16);
            } else {
                dec_lag3(index, sf == 0, &mut t0, &mut t0_frac);
                self.old_t0 = t0;
            }
            t2[sf] = t0;

            pred_lt_3(&mut self.old_exc, pos, t0, t0_frac, L_SUBFR);

            // Fixed codebook, random on a bad frame.
            if bfi {
                prm[p] = self.random.draw() & 0x1fff;
                prm[p + 1] = self.random.draw() & 0x000f;
            }
            let mut code = [0i16; L_SUBFR];
            decod_acelp(prm[p + 1], prm[p], &mut code);
            p += 2;

            let j = shl(self.sharp, 1);
            if (t0 as usize) < L_SUBFR {
                for i in t0 as usize..L_SUBFR {
                    code[i] = add(code[i], mult(code[i - t0 as usize], j));
                }
            }

            let index = prm[p];
            p += 1;
            self.gain
                .decode(index, &code, bfi, &mut self.gain_pitch, &mut self.gain_code);

            self.sharp = self.gain_pitch.clamp(SHARPMIN, SHARPMAX);

            for i in 0..L_SUBFR {
                let mut t = l_mult(self.old_exc[pos + i], self.gain_pitch);
                t = l_mac(t, code[i], self.gain_code);
                t = l_shl(t, 1);
                self.old_exc[pos + i] = round(t);
            }

            // Synthesis; on overflow scale the whole excitation history
            // down and synthesise again.
            let out = &mut synth[i_subfr..i_subfr + L_SUBFR];
            clear_overflow();
            let mut mem = self.mem_syn;
            syn_filt(az, &self.old_exc[pos..pos + L_SUBFR], out, &mut mem, false);
            if overflow() {
                for v in &mut self.old_exc {
                    *v = shr(*v, 2);
                }
                syn_filt(
                    az,
                    &self.old_exc[pos..pos + L_SUBFR],
                    out,
                    &mut self.mem_syn,
                    true,
                );
            } else {
                self.mem_syn.copy_from_slice(&out[L_SUBFR - M..]);
            }
        }

        self.old_exc.copy_within(L_FRAME.., 0);

        self.post_filter.run(&mut synth, &a_t, t2);
        self.post_process.run(&mut synth);
        synth
    }
}

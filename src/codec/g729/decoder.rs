//! The G.729 Annex A decoder with Annex B comfort noise (DECODER.C,
//! DEC_LD8A.C), postfilter and output filter included.

use super::acelp::decod_acelp;
use super::basic_op::{
    add, clear_overflow, l_mac, l_mult, l_shl, mult, norm_l, overflow, round, shl, shr, sub,
};
use super::dsp::{PostProcess, Random, syn_filt};
use super::dtx::{CngDecodeContext, CngDecoder, INIT_SEED};
use super::gain::GainDecoder;
use super::lpc::int_qlpc;
use super::lsp::LspDecoder;
use super::pitch::{EXC_HISTORY, check_parity_pitch, dec_lag3, pred_lt_3};
use super::postfilter::PostFilter;
use super::{L_FRAME, L_SUBFR, LSP_INIT, M, MP1, PIT_MAX, PRM_SIZE, SHARPMAX, SHARPMIN};

/// Start of the current frame in the excitation buffer.
const EXC: usize = EXC_HISTORY;

/// One received frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frame {
    /// Active speech: the 11 parameters.
    Speech([i16; PRM_SIZE]),
    /// Annex B silence insertion descriptor: the 4 parameters.
    Sid([i16; 4]),
    /// Annex B: nothing was sent (DTX).
    NotSent,
}

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
    /// Random codebook indices for erased speech frames.
    seed_fer: Random,
    /// Comfort-noise generator, reset on every speech frame.
    seed: Random,
    cng: CngDecoder,
    /// Whether the previous frame was speech (as opposed to comfort noise).
    past_speech: bool,
    /// Energy of the last good frame's excitation, for a SID-less
    /// transition into comfort noise.
    sid_sav: i16,
    sh_sid_sav: i16,
    post_filter: PostFilter,
    post_process: PostProcess,
}

impl Decoder {
    pub fn new() -> Self {
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
            seed_fer: Random::new(),
            seed: Random::with_seed(INIT_SEED),
            cng: CngDecoder::new(),
            past_speech: true,
            sid_sav: 0,
            sh_sid_sav: 1,
            post_filter: PostFilter::new(),
            post_process: PostProcess::new(),
        }
    }

    /// Whether the decoder is in a talkspurt (as opposed to comfort noise).
    pub fn in_speech(&self) -> bool {
        self.past_speech
    }

    /// Decode one frame to 80 samples. With `bfi` (bad frame indicator) the
    /// frame's contents are ignored: an erased frame during speech is
    /// concealed as speech, one during comfort noise continues the noise.
    pub fn decode_frame(&mut self, frame: &Frame, bfi: bool) -> [i16; L_FRAME] {
        let speech = if bfi {
            self.past_speech
        } else {
            matches!(frame, Frame::Speech(_))
        };

        let mut synth = [0i16; L_FRAME];
        let mut a_t = [0i16; 2 * MP1];
        let mut t2 = [0i16; 2];

        if speech {
            self.seed = Random::with_seed(INIT_SEED);
            let prm = match frame {
                Frame::Speech(p) if !bfi => *p,
                _ => [0; PRM_SIZE],
            };
            self.decode_speech(&prm, bfi, &mut synth, &mut a_t, &mut t2);
        } else {
            let sid = match frame {
                Frame::Sid(p) if !bfi => Some(p),
                _ => None,
            };
            self.cng.decode(CngDecodeContext {
                past_speech: self.past_speech,
                sid_sav: self.sid_sav,
                sh_sid_sav: self.sh_sid_sav,
                sid,
                exc: &mut self.old_exc,
                pos: EXC,
                lsp_old: &mut self.lsp_old,
                a_t: &mut a_t,
                seed: &mut self.seed,
                freq_prev: self.lsp.freq_prev_mut(),
            });
            for (sf, t) in t2.iter_mut().enumerate() {
                self.synthesise(&a_t, sf, &mut synth);
                *t = self.old_t0;
            }
            self.sharp = SHARPMIN;
        }

        if !bfi {
            let mut t = 0;
            for &v in &self.old_exc[EXC..EXC + L_FRAME] {
                t = l_mac(t, v, v);
            }
            let sh = norm_l(t);
            self.sid_sav = round(l_shl(t, sh));
            self.sh_sid_sav = sub(16, sh);
        }

        self.old_exc.copy_within(L_FRAME.., 0);
        self.past_speech = speech;

        self.post_filter.run(&mut synth, &a_t, t2, speech);
        self.post_process.run(&mut synth);
        synth
    }

    fn decode_speech(
        &mut self,
        prm: &[i16; PRM_SIZE],
        bfi: bool,
        synth: &mut [i16; L_FRAME],
        a_t: &mut [i16; 2 * MP1],
        t2: &mut [i16; 2],
    ) {
        let mut prm = *prm;
        prm[3] = check_parity_pitch(prm[2], prm[3]);

        let mut lsp_new = [0i16; M];
        self.lsp.decode(&prm[..2], &mut lsp_new, bfi);
        int_qlpc(&self.lsp_old, &lsp_new, a_t);
        self.lsp_old = lsp_new;

        let mut t0 = 0;
        let mut t0_frac = 0;
        let mut p = 2;

        for (sf, t2_sf) in t2.iter_mut().enumerate() {
            let pos = EXC + sf * L_SUBFR;

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
            *t2_sf = t0;

            pred_lt_3(&mut self.old_exc, pos, t0, t0_frac, L_SUBFR);

            // Fixed codebook, random on a bad frame.
            if bfi {
                prm[p] = self.seed_fer.draw() & 0x1fff;
                prm[p + 1] = self.seed_fer.draw() & 0x000f;
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

            self.synthesise(a_t, sf, synth);
        }
    }

    /// Synthesis filter for subframe `sf`; on overflow scale the whole
    /// excitation history down and synthesise again.
    fn synthesise(&mut self, a_t: &[i16; 2 * MP1], sf: usize, synth: &mut [i16; L_FRAME]) {
        let az = &a_t[sf * MP1..(sf + 1) * MP1];
        let i_subfr = sf * L_SUBFR;
        let pos = EXC + i_subfr;
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
}

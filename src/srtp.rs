//! SRTP (Secure RTP) implementation per RFC 3711.
//!
//! Provides AES-128-CM encryption with HMAC-SHA1-80 authentication for
//! securing RTP and RTCP media streams using SDES key exchange (RFC 4568).
//!
//! # Example
//!
//! ```
//! use rtp_engine::srtp::SrtpContext;
//!
//! // Generate keying material
//! let (mut sender, key_material) = SrtpContext::generate().unwrap();
//! let mut receiver = SrtpContext::from_base64(&key_material).unwrap();
//!
//! // Encrypt RTP packet
//! let rtp = vec![0x80, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xA0,
//!                0x12, 0x34, 0x56, 0x78, 0xDE, 0xAD, 0xBE, 0xEF];
//! let srtp = sender.protect_rtp(&rtp).unwrap();
//!
//! // Decrypt on receiver side
//! let decrypted = receiver.unprotect_rtp(&srtp).unwrap();
//! assert_eq!(decrypted, rtp);
//! ```

use aes::cipher::{KeyIvInit, StreamCipher};
use hmac::{Hmac, Mac};
use sha1::Sha1;

use crate::error::{Error, Result};

type Aes128Ctr = ctr::Ctr128BE<aes::Aes128>;
type HmacSha1 = Hmac<Sha1>;

const AUTH_TAG_LEN: usize = 10;
const MASTER_KEY_LEN: usize = 16;
const MASTER_SALT_LEN: usize = 14;

/// Total keying material length (master key + master salt).
pub const KEYING_MATERIAL_LEN: usize = MASTER_KEY_LEN + MASTER_SALT_LEN;

const SESSION_KEY_LEN: usize = 16;
const SESSION_SALT_LEN: usize = 14;
const SESSION_AUTH_KEY_LEN: usize = 20;

const LABEL_RTP_CIPHER: u8 = 0x00;
const LABEL_RTP_AUTH: u8 = 0x01;
const LABEL_RTP_SALT: u8 = 0x02;
const LABEL_RTCP_CIPHER: u8 = 0x03;
const LABEL_RTCP_AUTH: u8 = 0x04;
const LABEL_RTCP_SALT: u8 = 0x05;

/// How many sources' receive state a context remembers. A call has one remote
/// SSRC, occasionally two across a re-INVITE or a PBX re-anchoring the media;
/// the cap bounds memory against a peer that keeps changing it.
const MAX_RX_SOURCES: usize = 4;

/// The replay window size in packets (RFC 3711 §3.3.2 asks for at least 64).
const REPLAY_WINDOW: u64 = 64;

/// The sliding replay window of RFC 3711 §3.3.2 over one source's packet
/// indices: the highest index authenticated so far, and a bitmap of which of
/// the 64 indices at and below it have been seen.
#[derive(Debug, Clone, Copy, Default)]
struct ReplayWindow {
    highest: Option<u64>,
    seen: u64,
}

impl ReplayWindow {
    /// Whether `index` may be accepted: newer than anything seen, or inside
    /// the window and not yet seen.
    fn admits(&self, index: u64) -> bool {
        let Some(highest) = self.highest else {
            return true;
        };
        if index > highest {
            return true;
        }
        let behind = highest - index;
        behind < REPLAY_WINDOW && self.seen & (1 << behind) == 0
    }

    /// Record `index` as received. Only called after it authenticated.
    fn accept(&mut self, index: u64) {
        match self.highest {
            Some(highest) if index <= highest => self.seen |= 1 << (highest - index),
            Some(highest) => {
                let ahead = index - highest;
                self.seen = if ahead >= REPLAY_WINDOW {
                    0
                } else {
                    self.seen << ahead
                };
                self.seen |= 1;
                self.highest = Some(index);
            }
            None => {
                self.seen = 1;
                self.highest = Some(index);
            }
        }
    }
}

/// The rollover counter and sequence number this context sends with.
#[derive(Debug, Clone, Copy, Default)]
struct SendState {
    roc: u32,
    s_l: u16,
    initialized: bool,
}

/// What a receiver knows about one remote source: the ROC and sequence number
/// of the highest packet index it has authenticated, and the replay window.
#[derive(Debug, Clone, Copy)]
struct RecvState {
    ssrc: u32,
    roc: u32,
    s_l: u16,
    window: ReplayWindow,
}

impl RecvState {
    fn new(ssrc: u32) -> Self {
        Self {
            ssrc,
            roc: 0,
            s_l: 0,
            window: ReplayWindow::default(),
        }
    }

    /// The ROC a packet with sequence number `seq` most likely carries
    /// (RFC 3711 Appendix A), judged against the highest index seen.
    fn estimate_roc(&self, seq: u16) -> u32 {
        if self.window.highest.is_none() {
            return self.roc;
        }
        let diff = i32::from(seq) - i32::from(self.s_l);
        if diff > 0x8000 {
            // Numerically far ahead, so really a late packet from before the
            // last wrap.
            self.roc.wrapping_sub(1)
        } else if diff < -0x8000 {
            // Numerically far behind, so the sequence has wrapped.
            self.roc.wrapping_add(1)
        } else {
            self.roc
        }
    }
}

/// SRTP cryptographic context for a single media session.
///
/// Maintains session keys, the rollover counter for sending, and per-source
/// receive state with a replay window (RFC 3711 §3.3.2).
pub struct SrtpContext {
    session_key: [u8; SESSION_KEY_LEN],
    session_salt: [u8; SESSION_SALT_LEN],
    session_auth_key: [u8; SESSION_AUTH_KEY_LEN],
    rtcp_session_key: [u8; SESSION_KEY_LEN],
    rtcp_session_salt: [u8; SESSION_SALT_LEN],
    rtcp_session_auth_key: [u8; SESSION_AUTH_KEY_LEN],
    tx: SendState,
    /// Receive state per remote SSRC, most recently used first.
    rx: Vec<RecvState>,
    srtcp_index: u32,
    /// SRTCP replay windows per remote SSRC, most recently used first.
    rtcp_rx: Vec<(u32, ReplayWindow)>,
}

/// The length of an RTP header including CSRCs and any header extension
/// (RFC 3550 §5.3.1), or `None` when the packet is too short to hold it.
fn rtp_header_len(packet: &[u8]) -> Option<usize> {
    let cc = usize::from(*packet.first()? & 0x0F);
    let mut len = 12 + cc * 4;
    if packet[0] & 0x10 != 0 {
        let words = packet.get(len + 2..len + 4)?;
        len += 4 + usize::from(u16::from_be_bytes([words[0], words[1]])) * 4;
    }
    (packet.len() >= len).then_some(len)
}

impl SrtpContext {
    /// Create a new SRTP context from master key and master salt.
    ///
    /// The master key must be 16 bytes and master salt must be 14 bytes
    /// (AES_CM_128_HMAC_SHA1_80 profile).
    pub fn new(master_key: &[u8], master_salt: &[u8]) -> Result<Self> {
        if master_key.len() != MASTER_KEY_LEN {
            return Err(Error::srtp(format!(
                "master key must be {} bytes, got {}",
                MASTER_KEY_LEN,
                master_key.len()
            )));
        }
        if master_salt.len() != MASTER_SALT_LEN {
            return Err(Error::srtp(format!(
                "master salt must be {} bytes, got {}",
                MASTER_SALT_LEN,
                master_salt.len()
            )));
        }

        let session_key = derive_session_key(master_key, master_salt, LABEL_RTP_CIPHER)?;
        let session_salt = derive_session_salt(master_key, master_salt, LABEL_RTP_SALT)?;
        let session_auth_key = derive_session_auth_key(master_key, master_salt, LABEL_RTP_AUTH)?;

        let rtcp_session_key = derive_session_key(master_key, master_salt, LABEL_RTCP_CIPHER)?;
        let rtcp_session_salt = derive_session_salt(master_key, master_salt, LABEL_RTCP_SALT)?;
        let rtcp_session_auth_key =
            derive_session_auth_key(master_key, master_salt, LABEL_RTCP_AUTH)?;

        Ok(Self {
            session_key,
            session_salt,
            session_auth_key,
            rtcp_session_key,
            rtcp_session_salt,
            rtcp_session_auth_key,
            tx: SendState::default(),
            rx: Vec::new(),
            srtcp_index: 0,
            rtcp_rx: Vec::new(),
        })
    }

    /// Create from base64-encoded keying material (as used in SDP `a=crypto`).
    pub fn from_base64(b64_key_material: &str) -> Result<Self> {
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64_key_material.trim())
            .map_err(|e| Error::srtp(format!("base64 decode error: {}", e)))?;
        if decoded.len() < KEYING_MATERIAL_LEN {
            return Err(Error::srtp(format!(
                "keying material too short: {} bytes, need {}",
                decoded.len(),
                KEYING_MATERIAL_LEN
            )));
        }
        Self::new(
            &decoded[..MASTER_KEY_LEN],
            &decoded[MASTER_KEY_LEN..KEYING_MATERIAL_LEN],
        )
    }

    /// Generate random keying material and return (context, base64-key).
    pub fn generate() -> Result<(Self, String)> {
        use base64::Engine;
        let material: [u8; KEYING_MATERIAL_LEN] = rand::random();
        let b64 = base64::engine::general_purpose::STANDARD.encode(material);
        let ctx = Self::new(&material[..MASTER_KEY_LEN], &material[MASTER_KEY_LEN..])?;
        Ok((ctx, b64))
    }

    /// Encrypt and authenticate an RTP packet.
    pub fn protect_rtp(&mut self, rtp_packet: &[u8]) -> Result<Vec<u8>> {
        if rtp_packet.len() < 12 {
            return Err(Error::rtp("RTP packet too short"));
        }
        let header_len =
            rtp_header_len(rtp_packet).ok_or_else(|| Error::rtp("RTP packet truncated"))?;

        let seq = u16::from_be_bytes([rtp_packet[2], rtp_packet[3]]);
        let ssrc =
            u32::from_be_bytes([rtp_packet[8], rtp_packet[9], rtp_packet[10], rtp_packet[11]]);

        // Update ROC for sending
        if !self.tx.initialized {
            self.tx.initialized = true;
            log::debug!("SRTP TX: first packet, seq={seq}, ssrc={ssrc}");
        } else if seq == 0 && self.tx.s_l == 0xFFFF {
            self.tx.roc = self.tx.roc.wrapping_add(1);
        }
        self.tx.s_l = seq;

        let iv = build_iv(&self.session_salt, ssrc, self.tx.roc, seq);

        let mut output = rtp_packet.to_vec();
        aes_128_cm_encrypt(&self.session_key, &iv, &mut output[header_len..])?;

        let tag = compute_rtp_auth_tag(&self.session_auth_key, &output, self.tx.roc)?;
        output.extend_from_slice(&tag);

        Ok(output)
    }

    /// Verify and decrypt an SRTP packet.
    ///
    /// Fails with [`Error::Srtp`] when the packet is malformed or does not
    /// authenticate, and with [`Error::SrtpReplay`] when it authenticates but
    /// was already received or is older than the replay window. Neither moves
    /// the receive state.
    pub fn unprotect_rtp(&mut self, srtp_packet: &[u8]) -> Result<Vec<u8>> {
        if srtp_packet.len() < 12 + AUTH_TAG_LEN {
            return Err(Error::srtp("SRTP packet too short"));
        }

        let auth_boundary = srtp_packet.len() - AUTH_TAG_LEN;
        let authenticated_portion = &srtp_packet[..auth_boundary];
        let received_tag = &srtp_packet[auth_boundary..];
        let header_len = rtp_header_len(authenticated_portion)
            .ok_or_else(|| Error::srtp("SRTP packet truncated"))?;

        let seq = u16::from_be_bytes([srtp_packet[2], srtp_packet[3]]);
        let ssrc = u32::from_be_bytes([
            srtp_packet[8],
            srtp_packet[9],
            srtp_packet[10],
            srtp_packet[11],
        ]);

        let position = self.rx.iter().position(|state| state.ssrc == ssrc);
        let state = position.map_or_else(|| RecvState::new(ssrc), |i| self.rx[i]);
        let estimated_roc = state.estimate_roc(seq);

        let computed_tag =
            compute_rtp_auth_tag(&self.session_auth_key, authenticated_portion, estimated_roc)?;
        if !constant_time_eq(&computed_tag, received_tag) {
            // Debug, not warn: a peer or an attacker sending garbage at line
            // rate would otherwise flood the log. Callers count these.
            log::debug!("SRTP RX: authentication failed, seq={seq}, ssrc={ssrc}");
            return Err(Error::srtp("SRTP authentication failed"));
        }

        let index = (u64::from(estimated_roc) << 16) | u64::from(seq);
        if !state.window.admits(index) {
            log::debug!("SRTP RX: replayed or stale packet dropped, seq={seq}, ssrc={ssrc}");
            return Err(Error::SrtpReplay);
        }

        let mut state = state;
        if state.window.highest.is_none_or(|highest| index > highest) {
            state.roc = estimated_roc;
            state.s_l = seq;
        }
        state.window.accept(index);
        if position.is_none() {
            log::debug!("SRTP RX: first packet from ssrc={ssrc}, seq={seq}");
        }
        self.remember_source(position, state);

        let iv = build_iv(&self.session_salt, ssrc, estimated_roc, seq);
        let mut output = authenticated_portion.to_vec();
        aes_128_cm_encrypt(&self.session_key, &iv, &mut output[header_len..])?;

        Ok(output)
    }

    /// Store a source's receive state at the front of the list, evicting the
    /// least recently used source beyond [`MAX_RX_SOURCES`].
    fn remember_source(&mut self, position: Option<usize>, state: RecvState) {
        if let Some(i) = position {
            self.rx.remove(i);
        }
        self.rx.insert(0, state);
        self.rx.truncate(MAX_RX_SOURCES);
    }

    /// Encrypt and authenticate an RTCP packet.
    pub fn protect_rtcp(&mut self, rtcp_packet: &[u8]) -> Result<Vec<u8>> {
        if rtcp_packet.len() < 8 {
            return Err(Error::rtcp("RTCP packet too short"));
        }

        let ssrc = u32::from_be_bytes([
            rtcp_packet[4],
            rtcp_packet[5],
            rtcp_packet[6],
            rtcp_packet[7],
        ]);

        let index = self.srtcp_index;
        self.srtcp_index = self.srtcp_index.wrapping_add(1) & 0x7FFF_FFFF;

        let iv = build_rtcp_iv(&self.rtcp_session_salt, ssrc, index);

        let mut output = rtcp_packet.to_vec();
        if output.len() > 8 {
            aes_128_cm_encrypt(&self.rtcp_session_key, &iv, &mut output[8..])?;
        }

        let e_index = index | 0x8000_0000;
        output.extend_from_slice(&e_index.to_be_bytes());

        let tag = compute_rtcp_auth_tag(&self.rtcp_session_auth_key, &output)?;
        output.extend_from_slice(&tag);

        Ok(output)
    }

    /// Verify and decrypt an SRTCP packet.
    pub fn unprotect_rtcp(&mut self, srtcp_packet: &[u8]) -> Result<Vec<u8>> {
        if srtcp_packet.len() < 8 + 4 + AUTH_TAG_LEN {
            return Err(Error::srtp("SRTCP packet too short"));
        }

        let auth_boundary = srtcp_packet.len() - AUTH_TAG_LEN;
        let authenticated_portion = &srtcp_packet[..auth_boundary];
        let received_tag = &srtcp_packet[auth_boundary..];

        let computed_tag =
            compute_rtcp_auth_tag(&self.rtcp_session_auth_key, authenticated_portion)?;
        if !constant_time_eq(&computed_tag, received_tag) {
            return Err(Error::srtp("SRTCP authentication failed"));
        }

        let index_offset = authenticated_portion.len() - 4;
        let e_index = u32::from_be_bytes([
            authenticated_portion[index_offset],
            authenticated_portion[index_offset + 1],
            authenticated_portion[index_offset + 2],
            authenticated_portion[index_offset + 3],
        ]);
        let encrypted = (e_index & 0x8000_0000) != 0;
        let index = e_index & 0x7FFF_FFFF;

        let ssrc = u32::from_be_bytes([
            authenticated_portion[4],
            authenticated_portion[5],
            authenticated_portion[6],
            authenticated_portion[7],
        ]);
        let position = self.rtcp_rx.iter().position(|(s, _)| *s == ssrc);
        let mut window = position.map_or_else(ReplayWindow::default, |i| self.rtcp_rx[i].1);
        if !window.admits(u64::from(index)) {
            return Err(Error::SrtpReplay);
        }
        window.accept(u64::from(index));
        if let Some(i) = position {
            self.rtcp_rx.remove(i);
        }
        self.rtcp_rx.insert(0, (ssrc, window));
        self.rtcp_rx.truncate(MAX_RX_SOURCES);

        let mut output = authenticated_portion[..index_offset].to_vec();

        if encrypted && output.len() > 8 {
            let ssrc = u32::from_be_bytes([output[4], output[5], output[6], output[7]]);
            let iv = build_rtcp_iv(&self.rtcp_session_salt, ssrc, index);
            aes_128_cm_encrypt(&self.rtcp_session_key, &iv, &mut output[8..])?;
        }

        Ok(output)
    }

    /// The receive ROC for `ssrc`, for tests.
    #[cfg(test)]
    fn rx_roc(&self, ssrc: u32) -> Option<u32> {
        self.rx
            .iter()
            .find(|state| state.ssrc == ssrc)
            .map(|state| state.roc)
    }
}

impl std::fmt::Debug for SrtpContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SrtpContext")
            .field("initialized", &self.tx.initialized)
            .field("tx_roc", &self.tx.roc)
            .field("rx_sources", &self.rx.len())
            .finish_non_exhaustive()
    }
}

fn build_iv(session_salt: &[u8; SESSION_SALT_LEN], ssrc: u32, roc: u32, seq: u16) -> [u8; 16] {
    // RFC 3711 Section 4.1:
    // IV = (k_s * 2^16) XOR (SSRC * 2^64) XOR (i * 2^16)
    // k_s (14 bytes) shifted left by 16 bits -> bytes [0..14]
    // SSRC (4 bytes) shifted left by 64 bits -> bytes [4..8]
    // i = ROC||SEQ (6 bytes) shifted left by 16 bits -> bytes [8..14]
    // bytes [14..16] = 0 (AES-CTR block counter)
    let mut iv = [0u8; 16];
    iv[4..8].copy_from_slice(&ssrc.to_be_bytes());
    iv[8..12].copy_from_slice(&roc.to_be_bytes());
    iv[12..14].copy_from_slice(&seq.to_be_bytes());
    for i in 0..SESSION_SALT_LEN {
        iv[i] ^= session_salt[i];
    }
    iv
}

fn build_rtcp_iv(session_salt: &[u8; SESSION_SALT_LEN], ssrc: u32, index: u32) -> [u8; 16] {
    // RFC 3711 Section 3.4:
    // IV = (k_s * 2^16) XOR (SSRC * 2^64) XOR (SRTCP index * 2^16)
    // Same layout as RTP IV but with SRTCP index instead of ROC||SEQ
    let mut iv = [0u8; 16];
    iv[4..8].copy_from_slice(&ssrc.to_be_bytes());
    iv[10..14].copy_from_slice(&index.to_be_bytes());
    for i in 0..SESSION_SALT_LEN {
        iv[i] ^= session_salt[i];
    }
    iv
}

fn aes_128_cm_encrypt(key: &[u8; 16], iv: &[u8; 16], data: &mut [u8]) -> Result<()> {
    let mut cipher = Aes128Ctr::new(key.into(), iv.into());
    cipher.apply_keystream(data);
    Ok(())
}

fn compute_rtp_auth_tag(
    auth_key: &[u8; SESSION_AUTH_KEY_LEN],
    authenticated_portion: &[u8],
    roc: u32,
) -> Result<[u8; AUTH_TAG_LEN]> {
    let mut mac = HmacSha1::new_from_slice(auth_key)
        .map_err(|e| Error::srtp(format!("HMAC init error: {}", e)))?;
    mac.update(authenticated_portion);
    mac.update(&roc.to_be_bytes());
    let result = mac.finalize().into_bytes();
    let mut tag = [0u8; AUTH_TAG_LEN];
    tag.copy_from_slice(&result[..AUTH_TAG_LEN]);
    Ok(tag)
}

fn compute_rtcp_auth_tag(
    auth_key: &[u8; SESSION_AUTH_KEY_LEN],
    authenticated_portion: &[u8],
) -> Result<[u8; AUTH_TAG_LEN]> {
    let mut mac = HmacSha1::new_from_slice(auth_key)
        .map_err(|e| Error::srtp(format!("HMAC init error: {}", e)))?;
    mac.update(authenticated_portion);
    let result = mac.finalize().into_bytes();
    let mut tag = [0u8; AUTH_TAG_LEN];
    tag.copy_from_slice(&result[..AUTH_TAG_LEN]);
    Ok(tag)
}

fn derive_session_key(
    master_key: &[u8],
    master_salt: &[u8],
    label: u8,
) -> Result<[u8; SESSION_KEY_LEN]> {
    let mut output = [0u8; SESSION_KEY_LEN];
    prf_aes_cm(master_key, master_salt, label, &mut output)?;
    Ok(output)
}

fn derive_session_salt(
    master_key: &[u8],
    master_salt: &[u8],
    label: u8,
) -> Result<[u8; SESSION_SALT_LEN]> {
    let mut output = [0u8; SESSION_SALT_LEN];
    prf_aes_cm(master_key, master_salt, label, &mut output)?;
    Ok(output)
}

fn derive_session_auth_key(
    master_key: &[u8],
    master_salt: &[u8],
    label: u8,
) -> Result<[u8; SESSION_AUTH_KEY_LEN]> {
    let mut output = [0u8; SESSION_AUTH_KEY_LEN];
    prf_aes_cm(master_key, master_salt, label, &mut output)?;
    Ok(output)
}

fn prf_aes_cm(master_key: &[u8], master_salt: &[u8], label: u8, output: &mut [u8]) -> Result<()> {
    let mut x = [0u8; 14];
    x[7] = label;

    for i in 0..14 {
        x[i] ^= master_salt[i];
    }

    let mut iv = [0u8; 16];
    iv[..14].copy_from_slice(&x);

    output.fill(0);
    let mut cipher = Aes128Ctr::new(master_key.into(), (&iv).into());
    cipher.apply_keystream(output);

    Ok(())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Parse an `a=crypto` attribute from SDP.
pub fn parse_sdp_crypto(sdp: &str) -> Option<String> {
    for line in sdp.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("a=crypto:") {
            let parts: Vec<&str> = rest.splitn(3, ' ').collect();
            if parts.len() >= 3
                && parts[1] == "AES_CM_128_HMAC_SHA1_80"
                && parts[2].starts_with("inline:")
            {
                let key_material = &parts[2]["inline:".len()..];
                let key_material = key_material.split('|').next().unwrap_or(key_material);
                return Some(key_material.to_string());
            }
        }
    }
    None
}

/// Build the `a=crypto` SDP attribute line.
pub fn build_sdp_crypto_line(b64_key: &str) -> String {
    format!("a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:{}\r\n", b64_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_rtp() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        let mut rtp = vec![0x80, 0x00];
        rtp.extend_from_slice(&1u16.to_be_bytes());
        rtp.extend_from_slice(&160u32.to_be_bytes());
        rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
        rtp.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let srtp = ctx_send.protect_rtp(&rtp).unwrap();
        assert_ne!(&srtp[12..12 + 4], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(srtp.len(), rtp.len() + AUTH_TAG_LEN);

        let decrypted = ctx_recv.unprotect_rtp(&srtp).unwrap();
        assert_eq!(decrypted, rtp);
    }

    #[test]
    fn test_roundtrip_rtcp() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        let mut rtcp = vec![0x80, 200];
        rtcp.extend_from_slice(&6u16.to_be_bytes());
        rtcp.extend_from_slice(&0xAABBCCDDu32.to_be_bytes());
        rtcp.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let srtcp = ctx_send.protect_rtcp(&rtcp).unwrap();
        let decrypted = ctx_recv.unprotect_rtcp(&srtcp).unwrap();
        assert_eq!(decrypted, rtcp);
    }

    #[test]
    fn test_auth_failure() {
        let (mut ctx_send, _) = SrtpContext::generate().unwrap();
        let (_, b64_other) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64_other).unwrap();

        let mut rtp = vec![0x80, 0x00];
        rtp.extend_from_slice(&1u16.to_be_bytes());
        rtp.extend_from_slice(&160u32.to_be_bytes());
        rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
        rtp.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let srtp = ctx_send.protect_rtp(&rtp).unwrap();
        assert!(ctx_recv.unprotect_rtp(&srtp).is_err());
    }

    #[test]
    fn test_parse_sdp_crypto() {
        let sdp = "v=0\r\n\
                   a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:dGVzdGtleXRlc3RrZXkxMjM0NTY3ODkwMTI=\r\n";
        let key = parse_sdp_crypto(sdp);
        assert_eq!(
            key,
            Some("dGVzdGtleXRlc3RrZXkxMjM0NTY3ODkwMTI=".to_string())
        );
    }

    #[test]
    fn test_parse_sdp_crypto_with_lifetime() {
        // SDP with MKI and lifetime parameters
        let sdp = "v=0\r\n\
                   a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:YUJjRGVmZ0hpSktsbU5PUHF|2^31\r\n";
        let key = parse_sdp_crypto(sdp);
        assert_eq!(key, Some("YUJjRGVmZ0hpSktsbU5PUHF".to_string()));
    }

    #[test]
    fn test_parse_sdp_crypto_no_crypto_line() {
        let sdp = "v=0\r\nm=audio 5004 RTP/AVP 0\r\n";
        assert!(parse_sdp_crypto(sdp).is_none());
    }

    #[test]
    fn test_parse_sdp_crypto_wrong_suite() {
        let sdp = "v=0\r\na=crypto:1 AES_CM_256_HMAC_SHA1_80 inline:key=\r\n";
        assert!(parse_sdp_crypto(sdp).is_none());
    }

    #[test]
    fn test_build_sdp_crypto_line() {
        let key = "YUJjRGVmZ0hpSktsbU5PUHF";
        let line = build_sdp_crypto_line(key);
        assert_eq!(
            line,
            "a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:YUJjRGVmZ0hpSktsbU5PUHF\r\n"
        );
    }

    #[test]
    fn test_srtp_sequence_rollover() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        // Start near rollover and go through it sequentially
        // This simulates a ~22 minute call at 50 packets/second
        let start_seq = 65530u16;
        for i in 0u16..12 {
            let seq = start_seq.wrapping_add(i);
            let ts = (start_seq as u32 + i as u32) * 160;

            let mut rtp = vec![0x80, 0x00];
            rtp.extend_from_slice(&seq.to_be_bytes());
            rtp.extend_from_slice(&ts.to_be_bytes());
            rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
            rtp.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

            let srtp = ctx_send.protect_rtp(&rtp).unwrap();
            let decrypted = ctx_recv.unprotect_rtp(&srtp).unwrap();
            assert_eq!(decrypted, rtp, "Failed at seq {} (i={})", seq, i);
        }
    }

    #[test]
    fn test_srtp_continuous_through_rollover() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        // Simulate continuous packet flow through a rollover boundary
        // Start near max and go through rollover (realistic 22+ minute call)
        let start = 65520u16;
        for i in 0u16..32 {
            let seq = start.wrapping_add(i);
            let ts = (start as u32 + i as u32) * 160;

            let mut rtp = vec![0x80, 0x00];
            rtp.extend_from_slice(&seq.to_be_bytes());
            rtp.extend_from_slice(&ts.to_be_bytes());
            rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
            rtp.extend_from_slice(&[(i & 0xFF) as u8; 4]);

            let srtp = ctx_send.protect_rtp(&rtp).unwrap();
            let decrypted = ctx_recv.unprotect_rtp(&srtp).unwrap();
            assert_eq!(decrypted, rtp, "Failed at seq {} (i={})", seq, i);
        }

        // Verify ROC state - sender should have incremented ROC
        assert_eq!(ctx_send.tx.roc, 1, "Sender ROC should be 1 after rollover");
        assert_eq!(
            ctx_recv.rx_roc(0x12345678),
            Some(1),
            "Receiver ROC should be 1 after rollover"
        );
    }

    #[test]
    fn test_srtp_second_rollover() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        // First rollover
        let start1 = 65530u16;
        for i in 0u16..12 {
            let seq = start1.wrapping_add(i);
            let ts = (start1 as u32 + i as u32) * 160;

            let mut rtp = vec![0x80, 0x00];
            rtp.extend_from_slice(&seq.to_be_bytes());
            rtp.extend_from_slice(&ts.to_be_bytes());
            rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
            rtp.extend_from_slice(&[0xAA; 4]);

            let srtp = ctx_send.protect_rtp(&rtp).unwrap();
            let decrypted = ctx_recv.unprotect_rtp(&srtp).unwrap();
            assert_eq!(decrypted, rtp, "First rollover failed at seq {}", seq);
        }

        assert_eq!(ctx_send.tx.roc, 1);
        assert_eq!(ctx_recv.rx_roc(0x12345678), Some(1));

        // Continue from where we left off (seq 6 after rollover from 65535 to 0)
        // Now go through second rollover
        let current_seq = start1.wrapping_add(12); // Should be 6
        let start2 = 65530u16;

        // Need to send packets continuously to get to second rollover
        // From seq 6 to seq 65530, then through to seq 6 again
        for seq in current_seq..start2 {
            let ts = (65530u32 + 12 + (seq - current_seq) as u32) * 160;

            let mut rtp = vec![0x80, 0x00];
            rtp.extend_from_slice(&seq.to_be_bytes());
            rtp.extend_from_slice(&ts.to_be_bytes());
            rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
            rtp.extend_from_slice(&[0xBB; 4]);

            let srtp = ctx_send.protect_rtp(&rtp).unwrap();
            let decrypted = ctx_recv.unprotect_rtp(&srtp).unwrap();
            assert_eq!(decrypted, rtp, "Gap packet failed at seq {}", seq);
        }

        // ROC should still be 1 (no rollover yet)
        assert_eq!(ctx_send.tx.roc, 1);
        assert_eq!(ctx_recv.rx_roc(0x12345678), Some(1));

        // Now second rollover: 65530 -> 65535 -> 0 -> 5
        for i in 0u16..12 {
            let seq = start2.wrapping_add(i);
            let ts = (65530u32 * 2 + i as u32) * 160;

            let mut rtp = vec![0x80, 0x00];
            rtp.extend_from_slice(&seq.to_be_bytes());
            rtp.extend_from_slice(&ts.to_be_bytes());
            rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
            rtp.extend_from_slice(&[0xCC; 4]);

            let srtp = ctx_send.protect_rtp(&rtp).unwrap();
            let decrypted = ctx_recv.unprotect_rtp(&srtp).unwrap();
            assert_eq!(decrypted, rtp, "Second rollover failed at seq {}", seq);
        }

        // After second rollover
        assert_eq!(
            ctx_send.tx.roc, 2,
            "Sender ROC should be 2 after second rollover"
        );
        assert_eq!(
            ctx_recv.rx_roc(0x12345678),
            Some(2),
            "Receiver ROC should be 2 after second rollover"
        );
    }

    #[test]
    fn test_srtp_out_of_order_near_rollover() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        // Build packets around rollover but send slightly out of order
        let sequences = [65534u16, 65535, 0, 1, 2, 3];
        let mut packets: Vec<(u16, Vec<u8>)> = Vec::new();

        for (i, &seq) in sequences.iter().enumerate() {
            let ts = (65534u32 + i as u32) * 160;
            let mut rtp = vec![0x80, 0x00];
            rtp.extend_from_slice(&seq.to_be_bytes());
            rtp.extend_from_slice(&ts.to_be_bytes());
            rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
            rtp.extend_from_slice(&[i as u8; 4]);

            let srtp = ctx_send.protect_rtp(&rtp).unwrap();
            packets.push((seq, srtp));
        }

        // Receive in slightly different order: 65534, 65535, 1, 0, 2, 3
        // (swap 0 and 1 to test reordering across rollover)
        let receive_order = [0, 1, 3, 2, 4, 5];
        for &idx in &receive_order {
            let (seq, ref srtp) = packets[idx];
            let result = ctx_recv.unprotect_rtp(srtp);
            assert!(
                result.is_ok(),
                "Failed to decrypt seq {} (idx {})",
                seq,
                idx
            );
        }
    }

    #[test]
    fn test_srtp_multiple_packets() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        // Send multiple packets
        for seq in 0u16..100 {
            let mut rtp = vec![0x80, 0x00];
            rtp.extend_from_slice(&seq.to_be_bytes());
            rtp.extend_from_slice(&((seq as u32) * 160).to_be_bytes());
            rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
            rtp.extend_from_slice(&[seq as u8; 160]);

            let srtp = ctx_send.protect_rtp(&rtp).unwrap();
            let decrypted = ctx_recv.unprotect_rtp(&srtp).unwrap();
            assert_eq!(decrypted, rtp);
        }
    }

    #[test]
    fn test_srtp_invalid_key_length() {
        // Too short master key
        let result = SrtpContext::new(&[0u8; 15], &[0u8; 14]);
        assert!(result.is_err());

        // Too short master salt
        let result = SrtpContext::new(&[0u8; 16], &[0u8; 13]);
        assert!(result.is_err());
    }

    #[test]
    fn test_srtp_invalid_base64() {
        let result = SrtpContext::from_base64("not-valid-base64!!!");
        assert!(result.is_err());
    }

    #[test]
    fn test_srtp_too_short_base64() {
        let result = SrtpContext::from_base64("YWJjZA=="); // Only 4 bytes
        assert!(result.is_err());
    }

    #[test]
    fn test_srtp_packet_too_short() {
        let (mut ctx, _) = SrtpContext::generate().unwrap();

        // RTP packet too short
        let short_rtp = vec![0x80, 0x00, 0x00, 0x01];
        assert!(ctx.protect_rtp(&short_rtp).is_err());

        // SRTP packet too short for auth tag
        let short_srtp = vec![
            0x80, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xA0, 0x12, 0x34, 0x56, 0x78,
        ];
        assert!(ctx.unprotect_rtp(&short_srtp).is_err());
    }

    #[test]
    fn test_srtp_tampered_packet() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        let mut rtp = vec![0x80, 0x00];
        rtp.extend_from_slice(&1u16.to_be_bytes());
        rtp.extend_from_slice(&160u32.to_be_bytes());
        rtp.extend_from_slice(&0x12345678u32.to_be_bytes());
        rtp.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let mut srtp = ctx_send.protect_rtp(&rtp).unwrap();

        // Tamper with payload
        srtp[12] ^= 0xFF;

        assert!(ctx_recv.unprotect_rtp(&srtp).is_err());
    }

    #[test]
    fn test_srtcp_multiple_packets() {
        let (mut ctx_send, b64) = SrtpContext::generate().unwrap();
        let mut ctx_recv = SrtpContext::from_base64(&b64).unwrap();

        // Send multiple RTCP packets
        for i in 0..10 {
            let mut rtcp = vec![0x80, 200];
            rtcp.extend_from_slice(&6u16.to_be_bytes());
            rtcp.extend_from_slice(&(0xAABBCC00u32 + i).to_be_bytes());
            rtcp.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

            let srtcp = ctx_send.protect_rtcp(&rtcp).unwrap();
            let decrypted = ctx_recv.unprotect_rtcp(&srtcp).unwrap();
            assert_eq!(decrypted, rtcp);
        }
    }

    #[test]
    fn test_rtcp_too_short() {
        let (mut ctx, _) = SrtpContext::generate().unwrap();

        // RTCP packet too short
        let short_rtcp = vec![0x80, 200, 0x00, 0x01];
        assert!(ctx.protect_rtcp(&short_rtcp).is_err());
    }

    #[test]
    fn test_srtp_context_debug() {
        let (ctx, _) = SrtpContext::generate().unwrap();
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("SrtpContext"));
        assert!(debug_str.contains("initialized"));
    }

    /// An RTP packet with the given SSRC and sequence number, its timestamp
    /// following the sequence at 20 ms of 8 kHz.
    fn rtp_packet(ssrc: u32, seq: u16) -> Vec<u8> {
        let mut rtp = vec![0x80, 0x00];
        rtp.extend_from_slice(&seq.to_be_bytes());
        rtp.extend_from_slice(&(u32::from(seq) * 160).to_be_bytes());
        rtp.extend_from_slice(&ssrc.to_be_bytes());
        rtp.extend_from_slice(&[seq as u8; 8]);
        rtp
    }

    fn pair() -> (SrtpContext, SrtpContext) {
        let (tx, b64) = SrtpContext::generate().unwrap();
        let rx = SrtpContext::from_base64(&b64).unwrap();
        (tx, rx)
    }

    /// RFC 3711 §3.3.2: a packet that authenticates but was already received
    /// must be dropped, or a captured stream can be played back into a call.
    #[test]
    fn a_replayed_rtp_packet_is_refused() {
        let (mut tx, mut rx) = pair();
        let packet = tx.protect_rtp(&rtp_packet(0x1234_5678, 100)).unwrap();
        assert!(rx.unprotect_rtp(&packet).is_ok());
        assert!(matches!(rx.unprotect_rtp(&packet), Err(Error::SrtpReplay)));
    }

    #[test]
    fn a_late_packet_inside_the_window_is_accepted_once() {
        let (mut tx, mut rx) = pair();
        let packets: Vec<Vec<u8>> = (0u16..40)
            .map(|seq| tx.protect_rtp(&rtp_packet(7, seq)).unwrap())
            .collect();
        for packet in &packets[20..] {
            rx.unprotect_rtp(packet).unwrap();
        }
        // 20 packets behind the newest: late, but inside the 64-packet window.
        assert!(rx.unprotect_rtp(&packets[5]).is_ok());
        assert!(matches!(
            rx.unprotect_rtp(&packets[5]),
            Err(Error::SrtpReplay)
        ));
    }

    #[test]
    fn a_packet_older_than_the_window_is_refused() {
        let (mut tx, mut rx) = pair();
        let old = tx.protect_rtp(&rtp_packet(7, 0)).unwrap();
        for seq in 1u16..=100 {
            rx.unprotect_rtp(&tx.protect_rtp(&rtp_packet(7, seq)).unwrap())
                .unwrap();
        }
        assert!(matches!(rx.unprotect_rtp(&old), Err(Error::SrtpReplay)));
    }

    /// A forged packet must not advance the window: if it did, one bogus
    /// packet with a high sequence number would make every real packet look
    /// stale.
    #[test]
    fn a_packet_that_fails_authentication_leaves_the_window_alone() {
        let (mut tx, mut rx) = pair();
        rx.unprotect_rtp(&tx.protect_rtp(&rtp_packet(7, 10)).unwrap())
            .unwrap();
        let mut forged = tx.protect_rtp(&rtp_packet(7, 30_000)).unwrap();
        let last = forged.len() - 1;
        forged[last] ^= 0x01;
        assert!(matches!(rx.unprotect_rtp(&forged), Err(Error::Srtp(_))));
        let next = tx.protect_rtp(&rtp_packet(7, 11)).unwrap();
        assert!(rx.unprotect_rtp(&next).is_ok());
    }

    /// A packet from just before a rollover, arriving after it, belongs to the
    /// previous ROC. Accepting it must not move the receiver back a cycle, or
    /// every packet after it fails authentication for the rest of the call.
    #[test]
    fn a_late_packet_does_not_rewind_the_rollover_counter() {
        let (mut tx, mut rx) = pair();
        let mut packets = Vec::new();
        for seq in [65_533u16, 65_534, 65_535, 0, 1, 2, 3] {
            packets.push(tx.protect_rtp(&rtp_packet(9, seq)).unwrap());
        }
        // 65533, 65535, 0, 1 arrive; 65534 is late.
        for index in [0, 2, 3, 4] {
            rx.unprotect_rtp(&packets[index]).unwrap();
        }
        rx.unprotect_rtp(&packets[1]).unwrap();
        rx.unprotect_rtp(&packets[5]).unwrap();
        rx.unprotect_rtp(&packets[6]).unwrap();
    }

    /// Each SSRC has its own index space (RFC 3711 §3.2.1), so a second
    /// source starting at a low sequence number is not "old", and switching
    /// back does not reopen the first source's window to replays.
    #[test]
    fn each_ssrc_keeps_its_own_replay_state() {
        let (mut tx, b64) = SrtpContext::generate().unwrap();
        let mut rx = SrtpContext::from_base64(&b64).unwrap();
        let first: Vec<Vec<u8>> = (1000u16..1010)
            .map(|seq| tx.protect_rtp(&rtp_packet(1, seq)).unwrap())
            .collect();
        for packet in &first {
            rx.unprotect_rtp(packet).unwrap();
        }
        let mut tx2 = SrtpContext::from_base64(&b64).unwrap();
        let second = tx2.protect_rtp(&rtp_packet(2, 5)).unwrap();
        assert!(rx.unprotect_rtp(&second).is_ok(), "a new SSRC starts fresh");
        assert!(matches!(
            rx.unprotect_rtp(&first[9]),
            Err(Error::SrtpReplay)
        ));
        assert!(matches!(rx.unprotect_rtp(&second), Err(Error::SrtpReplay)));
    }

    #[test]
    fn a_replayed_srtcp_packet_is_refused() {
        let (mut tx, mut rx) = pair();
        let mut rtcp = vec![0x80, 200];
        rtcp.extend_from_slice(&6u16.to_be_bytes());
        rtcp.extend_from_slice(&0xAABB_CCDDu32.to_be_bytes());
        rtcp.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let packet = tx.protect_rtcp(&rtcp).unwrap();
        assert!(rx.unprotect_rtcp(&packet).is_ok());
        assert!(matches!(rx.unprotect_rtcp(&packet), Err(Error::SrtpReplay)));
    }

    /// RFC 3711 §3.1: the header, extension included, is authenticated but not
    /// encrypted. Encrypting the extension garbles it for the far end.
    #[test]
    fn a_header_extension_is_authenticated_but_not_encrypted() {
        let (mut tx, mut rx) = pair();
        let mut rtp = vec![0x90, 0x00];
        rtp.extend_from_slice(&5u16.to_be_bytes());
        rtp.extend_from_slice(&800u32.to_be_bytes());
        rtp.extend_from_slice(&0x0102_0304u32.to_be_bytes());
        // One-byte header extension (RFC 8285), one 32-bit word.
        rtp.extend_from_slice(&[0xBE, 0xDE, 0x00, 0x01, 0x10, 0x7F, 0x00, 0x00]);
        rtp.extend_from_slice(&[0x55; 16]);
        let protected = tx.protect_rtp(&rtp).unwrap();
        assert_eq!(
            &protected[..20],
            &rtp[..20],
            "header and extension stay clear"
        );
        assert_ne!(&protected[20..36], &rtp[20..36], "the payload is encrypted");
        assert_eq!(rx.unprotect_rtp(&protected).unwrap(), rtp);

        let mut tampered = protected;
        tampered[17] ^= 0x01;
        assert!(
            rx.unprotect_rtp(&tampered).is_err(),
            "the extension is authenticated"
        );
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(&[1, 2, 3], &[1, 2, 3]));
        assert!(!constant_time_eq(&[1, 2, 3], &[1, 2, 4]));
        assert!(!constant_time_eq(&[1, 2, 3], &[1, 2]));
        assert!(constant_time_eq(&[], &[]));
    }
}

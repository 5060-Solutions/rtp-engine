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

/// SRTP cryptographic context for a single media session.
///
/// Maintains session keys and rollover counter for sequence number tracking.
pub struct SrtpContext {
    session_key: [u8; SESSION_KEY_LEN],
    session_salt: [u8; SESSION_SALT_LEN],
    session_auth_key: [u8; SESSION_AUTH_KEY_LEN],
    rtcp_session_key: [u8; SESSION_KEY_LEN],
    rtcp_session_salt: [u8; SESSION_SALT_LEN],
    rtcp_session_auth_key: [u8; SESSION_AUTH_KEY_LEN],
    roc: u32,
    s_l: u16,
    initialized: bool,
    srtcp_index: u32,
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
            roc: 0,
            s_l: 0,
            initialized: false,
            srtcp_index: 0,
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
        log::debug!(
            "SRTP from_base64: master_key[0..4]={:02x}{:02x}{:02x}{:02x}, salt[0..4]={:02x}{:02x}{:02x}{:02x}",
            decoded[0],
            decoded[1],
            decoded[2],
            decoded[3],
            decoded[MASTER_KEY_LEN],
            decoded[MASTER_KEY_LEN + 1],
            decoded[MASTER_KEY_LEN + 2],
            decoded[MASTER_KEY_LEN + 3]
        );
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

        let cc = (rtp_packet[0] & 0x0F) as usize;
        let header_len = 12 + cc * 4;
        if rtp_packet.len() < header_len {
            return Err(Error::rtp("RTP packet truncated"));
        }

        let seq = u16::from_be_bytes([rtp_packet[2], rtp_packet[3]]);
        let ssrc =
            u32::from_be_bytes([rtp_packet[8], rtp_packet[9], rtp_packet[10], rtp_packet[11]]);

        // Update ROC for sending
        let first_packet = !self.initialized;
        if !self.initialized {
            self.s_l = seq;
            self.initialized = true;
        } else if seq == 0 && self.s_l == 0xFFFF {
            self.roc = self.roc.wrapping_add(1);
        }
        self.s_l = seq;

        if first_packet {
            log::info!(
                "SRTP TX: first packet - seq={}, ssrc={}, roc={}, session_key[0..4]={:02x}{:02x}{:02x}{:02x}",
                seq,
                ssrc,
                self.roc,
                self.session_key[0],
                self.session_key[1],
                self.session_key[2],
                self.session_key[3]
            );
        }

        let iv = build_iv(&self.session_salt, ssrc, self.roc, seq);

        let mut output = rtp_packet.to_vec();
        aes_128_cm_encrypt(&self.session_key, &iv, &mut output[header_len..])?;

        let tag = compute_rtp_auth_tag(&self.session_auth_key, &output, self.roc)?;
        output.extend_from_slice(&tag);

        Ok(output)
    }

    /// Verify and decrypt an SRTP packet.
    pub fn unprotect_rtp(&mut self, srtp_packet: &[u8]) -> Result<Vec<u8>> {
        if srtp_packet.len() < 12 + AUTH_TAG_LEN {
            return Err(Error::srtp("SRTP packet too short"));
        }

        let cc = (srtp_packet[0] & 0x0F) as usize;
        let header_len = 12 + cc * 4;
        if srtp_packet.len() < header_len + AUTH_TAG_LEN {
            return Err(Error::srtp("SRTP packet truncated"));
        }

        let seq = u16::from_be_bytes([srtp_packet[2], srtp_packet[3]]);
        let ssrc = u32::from_be_bytes([
            srtp_packet[8],
            srtp_packet[9],
            srtp_packet[10],
            srtp_packet[11],
        ]);

        let first_packet = !self.initialized;
        let estimated_roc = self.estimate_roc(seq);

        let auth_boundary = srtp_packet.len() - AUTH_TAG_LEN;
        let authenticated_portion = &srtp_packet[..auth_boundary];
        let received_tag = &srtp_packet[auth_boundary..];

        let computed_tag =
            compute_rtp_auth_tag(&self.session_auth_key, authenticated_portion, estimated_roc)?;
        if !constant_time_eq(&computed_tag, received_tag) {
            log::warn!(
                "SRTP RX auth failed: seq={}, ssrc={}, roc={}, session_key[0..4]={:02x}{:02x}{:02x}{:02x}",
                seq,
                ssrc,
                estimated_roc,
                self.session_key[0],
                self.session_key[1],
                self.session_key[2],
                self.session_key[3]
            );
            return Err(Error::srtp("SRTP authentication failed"));
        }

        if first_packet {
            log::info!(
                "SRTP RX: first packet - seq={}, ssrc={}, roc={}, session_key[0..4]={:02x}{:02x}{:02x}{:02x}",
                seq,
                ssrc,
                estimated_roc,
                self.session_key[0],
                self.session_key[1],
                self.session_key[2],
                self.session_key[3]
            );
            // Don't set initialized here - let update_roc handle it so it
            // correctly initializes s_l before any estimate_roc calls.
        }

        self.update_roc(seq);

        let iv = build_iv(&self.session_salt, ssrc, estimated_roc, seq);
        let mut output = authenticated_portion.to_vec();
        aes_128_cm_encrypt(&self.session_key, &iv, &mut output[header_len..])?;

        Ok(output)
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

        let mut output = authenticated_portion[..index_offset].to_vec();

        if encrypted && output.len() > 8 {
            let ssrc = u32::from_be_bytes([output[4], output[5], output[6], output[7]]);
            let iv = build_rtcp_iv(&self.rtcp_session_salt, ssrc, index);
            aes_128_cm_encrypt(&self.rtcp_session_key, &iv, &mut output[8..])?;
        }

        Ok(output)
    }

    /// Estimate the ROC (Rollover Counter) for a received sequence number.
    ///
    /// RFC 3711 Appendix A: The algorithm uses signed arithmetic to detect
    /// whether a sequence number has wrapped around.
    fn estimate_roc(&self, seq: u16) -> u32 {
        if !self.initialized {
            return 0;
        }

        // Compute signed difference between seq and last seen sequence
        // This correctly handles the 16-bit wrap-around
        let diff = (seq as i32) - (self.s_l as i32);

        if diff > 0x8000 {
            // seq appears much larger than s_l, but it's actually a late packet
            // from before the current ROC (seq wrapped backward from our perspective)
            self.roc.wrapping_sub(1)
        } else if diff < -0x8000 {
            // s_l appears much larger than seq, meaning seq has rolled over
            // to a new ROC cycle (seq wrapped forward)
            self.roc.wrapping_add(1)
        } else {
            // Normal case: seq and s_l are within half the sequence space
            self.roc
        }
    }

    /// Update ROC state after successfully authenticating a packet.
    fn update_roc(&mut self, seq: u16) {
        if !self.initialized {
            self.s_l = seq;
            self.initialized = true;
            return;
        }

        let estimated = self.estimate_roc(seq);

        // Update ROC if it changed
        if estimated != self.roc {
            self.roc = estimated;
        }

        // Update s_l (highest sequence seen in current ROC)
        // Use signed difference to handle rollover edge case
        let diff = (seq as i32) - (self.s_l as i32);
        if diff > 0 || diff < -0x8000 {
            // seq > s_l normally, OR seq rolled over to new ROC
            self.s_l = seq;
        }
    }
}

impl std::fmt::Debug for SrtpContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SrtpContext")
            .field("initialized", &self.initialized)
            .field("roc", &self.roc)
            .field("s_l", &self.s_l)
            .finish()
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
        assert_eq!(ctx_send.roc, 1, "Sender ROC should be 1 after rollover");
        assert_eq!(ctx_recv.roc, 1, "Receiver ROC should be 1 after rollover");
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

        assert_eq!(ctx_send.roc, 1);
        assert_eq!(ctx_recv.roc, 1);

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
        assert_eq!(ctx_send.roc, 1);
        assert_eq!(ctx_recv.roc, 1);

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
            ctx_send.roc, 2,
            "Sender ROC should be 2 after second rollover"
        );
        assert_eq!(
            ctx_recv.roc, 2,
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

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(&[1, 2, 3], &[1, 2, 3]));
        assert!(!constant_time_eq(&[1, 2, 3], &[1, 2, 4]));
        assert!(!constant_time_eq(&[1, 2, 3], &[1, 2]));
        assert!(constant_time_eq(&[], &[]));
    }
}

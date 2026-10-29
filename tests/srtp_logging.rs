//! SRTP keys must never reach the log, at any level.
//!
//! Its own test binary because it installs the process-wide logger.
#![cfg(feature = "srtp")]

use std::sync::Mutex;

use base64::Engine;
use rtp_engine::srtp::SrtpContext;

static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if let Ok(mut lines) = LINES.lock() {
            lines.push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

static LOGGER: Capture = Capture;

fn rtp(seq: u16) -> Vec<u8> {
    let mut rtp = vec![0x80, 0x00];
    rtp.extend_from_slice(&seq.to_be_bytes());
    rtp.extend_from_slice(&160u32.to_be_bytes());
    rtp.extend_from_slice(&0x1234_5678u32.to_be_bytes());
    rtp.extend_from_slice(&[0xAB; 20]);
    rtp
}

#[test]
fn no_key_material_is_logged() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    let (mut tx, b64) = SrtpContext::generate().unwrap();
    let mut rx = SrtpContext::from_base64(&b64).unwrap();
    let (_, other) = SrtpContext::generate().unwrap();
    let mut wrong = SrtpContext::from_base64(&other).unwrap();
    for seq in 0..3 {
        let packet = tx.protect_rtp(&rtp(seq)).unwrap();
        rx.unprotect_rtp(&packet).unwrap();
        let _ = wrong.unprotect_rtp(&packet);
    }

    let material = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .unwrap();
    let hex = |bytes: &[u8]| {
        bytes.iter().fold(String::new(), |mut out, b| {
            use std::fmt::Write;
            let _ = write!(out, "{b:02x}");
            out
        })
    };
    let key_hex = hex(&material[..4]);
    let salt_hex = hex(&material[16..20]);
    let lines = LINES.lock().unwrap();
    for line in lines.iter() {
        assert!(!line.contains(&key_hex), "master key bytes logged: {line}");
        assert!(
            !line.contains(&salt_hex),
            "master salt bytes logged: {line}"
        );
        assert!(!line.contains(&b64), "keying material logged: {line}");
        assert!(!line.contains("session_key"), "session key logged: {line}");
    }
}

//! A media stream runs on the caller's task: no OS threads of its own, so a
//! host can carry hundreds of calls on a small runtime.
//!
//! Its own test binary, so no other test's threads come and go meanwhile.
#![cfg(feature = "stream")]

use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use rtp_engine::CodecType;
use rtp_engine::stream::{MediaStream, StreamConfig};

/// The process's OS thread count, where the platform says.
fn threads() -> Option<usize> {
    if cfg!(target_os = "linux") {
        return std::fs::read_dir("/proc/self/task")
            .ok()
            .map(Iterator::count);
    }
    if cfg!(target_os = "macos") {
        let pid = std::process::id().to_string();
        let out = std::process::Command::new("ps")
            .args(["-M", "-p", &pid])
            .output()
            .ok()?;
        // One header line, then one line per thread.
        return Some(String::from_utf8_lossy(&out.stdout).lines().count() - 1);
    }
    None
}

#[test]
fn stream_has_no_threads() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let Some(before) = threads() else {
        return;
    };
    let during = runtime.block_on(async {
        let local = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0);
        let (a, b) = (
            tokio::net::UdpSocket::bind(local).await.unwrap(),
            tokio::net::UdpSocket::bind(local).await.unwrap(),
        );
        let (a_addr, b_addr) = (a.local_addr().unwrap(), b.local_addr().unwrap());
        let mut a = MediaStream::new(a, StreamConfig::new(CodecType::Pcmu, b_addr), None).unwrap();
        let mut b = MediaStream::new(b, StreamConfig::new(CodecType::Pcmu, a_addr), None).unwrap();
        for _ in 0..10 {
            a.send_frame(&[0; 160]).await.unwrap();
            let _ = tokio::time::timeout(Duration::from_millis(30), b.recv_frame()).await;
        }
        threads()
    });
    assert_eq!(during, Some(before), "the stream started threads");
}

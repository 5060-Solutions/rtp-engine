//! Error types for rtp-engine.

use thiserror::Error;

/// Result type alias for rtp-engine operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in rtp-engine.
#[derive(Error, Debug)]
pub enum Error {
    /// Codec initialization or encoding/decoding error.
    #[error("codec error: {0}")]
    Codec(String),

    /// RTP packet parsing or construction error.
    #[error("RTP error: {0}")]
    Rtp(String),

    /// RTCP packet parsing or construction error.
    #[error("RTCP error: {0}")]
    Rtcp(String),

    /// SRTP encryption/decryption error.
    #[error("SRTP error: {0}")]
    Srtp(String),

    /// Audio device error.
    #[error("device error: {0}")]
    Device(String),

    /// Network I/O error.
    #[error("network error: {0}")]
    Network(#[from] std::io::Error),

    /// STUN protocol error.
    #[error("STUN error: {0}")]
    Stun(String),

    /// Invalid configuration or parameter.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

impl Error {
    /// Create a codec error.
    pub fn codec(msg: impl Into<String>) -> Self {
        Self::Codec(msg.into())
    }

    /// Create an RTP error.
    pub fn rtp(msg: impl Into<String>) -> Self {
        Self::Rtp(msg.into())
    }

    /// Create an RTCP error.
    pub fn rtcp(msg: impl Into<String>) -> Self {
        Self::Rtcp(msg.into())
    }

    /// Create an SRTP error.
    pub fn srtp(msg: impl Into<String>) -> Self {
        Self::Srtp(msg.into())
    }

    /// Create a device error.
    pub fn device(msg: impl Into<String>) -> Self {
        Self::Device(msg.into())
    }

    /// Create a STUN error.
    pub fn stun(msg: impl Into<String>) -> Self {
        Self::Stun(msg.into())
    }

    /// Create an invalid parameter error.
    pub fn invalid_parameter(msg: impl Into<String>) -> Self {
        Self::InvalidParameter(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_constructors() {
        let e = Error::codec("test codec error");
        assert!(matches!(e, Error::Codec(_)));
        assert_eq!(format!("{}", e), "codec error: test codec error");

        let e = Error::rtp("test rtp error");
        assert!(matches!(e, Error::Rtp(_)));
        assert_eq!(format!("{}", e), "RTP error: test rtp error");

        let e = Error::rtcp("test rtcp error");
        assert!(matches!(e, Error::Rtcp(_)));
        assert_eq!(format!("{}", e), "RTCP error: test rtcp error");

        let e = Error::srtp("test srtp error");
        assert!(matches!(e, Error::Srtp(_)));
        assert_eq!(format!("{}", e), "SRTP error: test srtp error");

        let e = Error::device("test device error");
        assert!(matches!(e, Error::Device(_)));
        assert_eq!(format!("{}", e), "device error: test device error");

        let e = Error::invalid_parameter("bad param");
        assert!(matches!(e, Error::InvalidParameter(_)));
        assert_eq!(format!("{}", e), "invalid parameter: bad param");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let e: Error = io_err.into();
        assert!(matches!(e, Error::Network(_)));
        assert!(format!("{}", e).contains("file not found"));
    }

    #[test]
    fn test_error_debug() {
        let e = Error::codec("debug test");
        let debug_str = format!("{:?}", e);
        assert!(debug_str.contains("Codec"));
        assert!(debug_str.contains("debug test"));
    }
}

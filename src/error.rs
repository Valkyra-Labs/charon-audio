//! Error types for Charon

use thiserror::Error;

/// Result type alias using CharonError
pub type Result<T> = std::result::Result<T, CharonError>;

/// Main error type for Charon operations
#[derive(Error, Debug)]
pub enum CharonError {
    #[error("Audio error: {0}")]
    Audio(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    Model(String),

    #[cfg(feature = "ort-backend")]
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[cfg(feature = "candle-backend")]
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Resampling error: {0}")]
    Resampling(String),

    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Not supported: {0}")]
    NotSupported(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<String> for CharonError {
    fn from(s: String) -> Self {
        CharonError::Unknown(s)
    }
}

impl From<&str> for CharonError {
    fn from(s: &str) -> Self {
        CharonError::Unknown(s.to_string())
    }
}

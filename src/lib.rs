//! # Charon
//!
//! Modern Rust music source separation library using state-of-the-art ML inference.
//!
//! Charon provides a complete, pure-Rust implementation for audio source separation,
//! inspired by Demucs but built with modern Rust ML frameworks (ONNX Runtime via `ort`,
//! and HuggingFace Candle).
//!
//! ## Features
//!
//! - **Multiple ML Backends**: Support for ONNX Runtime (production-ready, hardware accelerated)
//!   and Candle (pure Rust, flexible)
//! - **Audio Processing**: Complete audio I/O with Symphonia (decode any format),
//!   Rubato (high-quality resampling), and Hound (WAV encoding)
//! - **Real-time Processing**: Support for real-time audio separation using CPAL
//! - **Hardware Acceleration**: CUDA, TensorRT, Metal, Accelerate support
//! - **Parallel Processing**: Multi-threaded audio processing with Rayon
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use charon::{Separator, SeparatorConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create a separator with default settings
//! let separator = Separator::new(SeparatorConfig::default())?;
//!
//! // Separate an audio file
//! let stems = separator.separate_file("input.mp3")?;
//!
//! // Save individual stems
//! stems.save_all("output_dir")?;
//! # Ok(())
//! # }
//! ```

pub mod audio;
pub mod error;
pub mod models;
pub mod model_zoo;
pub mod performance;
pub mod processor;
pub mod realtime;
pub mod separator;
pub mod utils;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export main types
pub use audio::{AudioBuffer, AudioFile, AudioFormat};
pub use error::{CharonError, Result};
pub use models::{ModelBackend, ModelConfig};
pub use model_zoo::{ModelMetadata, ModelZoo};
pub use performance::{AudioKNN, BatchProcessor, PerformanceHint, PerformanceHints, SimdOps};
pub use processor::{ProcessConfig, Processor};
pub use realtime::RealtimeSeparator;
pub use separator::{Separator, SeparatorConfig, Stems};

#[cfg(target_arch = "wasm32")]
pub use wasm::WasmSeparator;

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        assert_eq!(env!("CARGO_PKG_VERSION"), "0.1.0");
    }
}

//! # Charon
//!
//! Rust music source separation pipeline for ONNX models.
//!
//! Status: model inference is not implemented yet in this release; see the
//! README status section.
//!
//! ## Features
//!
//! - **ML backend**: ONNX Runtime via `ort` (`ort-backend` feature, on by default)
//! - **Audio I/O**: decoding with Symphonia, resampling with Rubato, WAV output with Hound
//! - **Parallel processing**: segments are processed with Rayon
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use charon_audio::{Separator, SeparatorConfig};
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
pub mod model_zoo;
pub mod models;
pub mod performance;
pub mod processor;
#[cfg(feature = "realtime")]
pub mod realtime;
pub mod separator;
pub mod utils;

// Re-export main types
pub use audio::{AudioBuffer, AudioFile, AudioFormat};
pub use error::{CharonError, Result};
pub use model_zoo::{ModelMetadata, ModelZoo};
pub use models::{ModelBackend, ModelConfig};
pub use performance::{AudioKNN, BatchProcessor, PerformanceHint, PerformanceHints, SimdOps};
pub use processor::{ProcessConfig, Processor};
#[cfg(feature = "realtime")]
pub use realtime::RealtimeSeparator;
pub use separator::{Separator, SeparatorConfig, Stems};

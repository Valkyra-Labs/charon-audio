//! # Charon
//!
//! Rust music source separation pipeline for ONNX models.
//!
//! Supported model: the 4-stem HTDemucs ONNX export
//! ([`SeparatorConfig::htdemucs`]). Segmentation, overlap-add and
//! normalization follow Demucs 4.1.0.
//!
//! ## Features
//!
//! - **ML backend**: ONNX Runtime via `ort` (`ort-backend` feature, on by default)
//! - **Audio I/O**: decoding with Symphonia (`decode` feature; AAC with `aac`), resampling
//!   with Rubato, WAV output with Hound, FLAC output with flacenc
//! - **Parallel processing**: segments are processed with Rayon
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use charon_audio::{Separator, SeparatorConfig};
//!
//! # #[cfg(feature = "decode")]
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
//! # #[cfg(not(feature = "decode"))]
//! # fn main() {}
//! ```

pub mod audio;
pub mod control;
pub mod error;
pub mod model_zoo;
pub mod models;
pub mod performance;
pub mod processor;
#[cfg(feature = "realtime")]
pub mod realtime;
pub mod regions;
pub mod separator;
pub mod stft;
pub mod stream;
pub mod utils;

// Re-export main types
pub use audio::{AudioBuffer, AudioFile, AudioFormat, BitDepth};
pub use control::{CancelToken, Control, Progress};
pub use error::{CharonError, Result};
pub use model_zoo::{ModelMetadata, ModelZoo};
pub use models::{
    ExecutionProvider, ModelBackend, ModelConfig, ModelContract, OnnxOptions, OptimizationLevel,
};
#[allow(deprecated)]
pub use performance::{AudioKNN, BatchProcessor, PerformanceHint, PerformanceHints, SimdOps};
pub use processor::{Blend, ProcessConfig, Processor};
#[cfg(feature = "realtime")]
pub use realtime::RealtimeSeparator;
pub use regions::{Region, RegionPlan, REGION_OUTPUTS};
pub use separator::{Separator, SeparatorConfig, StemFormat, Stems};
pub use stream::{AudioSource, BufferSink, BufferSource, StemSink};

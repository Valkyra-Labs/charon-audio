//! ML model backends and configuration

use crate::error::{CharonError, Result};
use ndarray::Array2;
#[cfg(feature = "ort-backend")]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
#[cfg(feature = "ort-backend")]
use std::sync::Mutex;

/// Model backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelBackend {
    /// ONNX Runtime
    #[cfg(feature = "ort-backend")]
    OnnxRuntime,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model file
    pub model_path: PathBuf,
    /// Model backend to use (optional, will be inferred if not set)
    #[serde(skip, default)]
    pub backend: Option<ModelBackend>,
    /// Expected sample rate
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: usize,
    /// Source names (e.g., ["drums", "bass", "vocals", "other"])
    pub sources: Vec<String>,
    /// Chunk size for processing (in samples)
    pub chunk_size: Option<usize>,
    /// Fixed model input length in samples, for models exported with a static
    /// time axis. Overrides `ProcessConfig::segment_length` when set.
    #[serde(default)]
    pub segment_samples: Option<usize>,
    /// Name of the model input tensor, shaped `[1, channels, samples]`
    #[serde(default = "default_input_name")]
    pub input_name: String,
    /// Name of the model output tensor, shaped `[1, sources, channels, samples]`
    #[serde(default = "default_output_name")]
    pub output_name: String,
    /// ONNX Runtime session options
    #[serde(default)]
    pub onnx: OnnxOptions,
}

fn default_input_name() -> String {
    "mix".to_string()
}

fn default_output_name() -> String {
    "stems".to_string()
}

/// Graph optimization level passed to ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OptimizationLevel {
    Disable,
    Basic,
    Extended,
    #[default]
    All,
}

/// Execution provider for ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    /// Apple CoreML (macOS/iOS), falls back to CPU for unsupported ops.
    /// Needs the `coreml` feature.
    CoreMl,
}

/// ONNX Runtime session options
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnnxOptions {
    /// Intra-op threads; `None` uses all available cores
    pub intra_threads: Option<usize>,
    pub optimization_level: OptimizationLevel,
    /// ORT memory-pattern optimization (pre-plans activation memory)
    pub memory_pattern: bool,
    /// ORT CPU memory arena
    pub cpu_arena: bool,
    /// Graph transformers to disable by name (for example `ConstantFolding`)
    pub disabled_optimizers: Vec<String>,
    pub execution_provider: ExecutionProvider,
}

impl Default for OnnxOptions {
    /// ONNX Runtime's own defaults
    fn default() -> Self {
        Self {
            intra_threads: None,
            optimization_level: OptimizationLevel::All,
            memory_pattern: true,
            cpu_arena: true,
            disabled_optimizers: Vec::new(),
            execution_provider: ExecutionProvider::Cpu,
        }
    }
}

impl OnnxOptions {
    /// Settings measured to cut peak memory on HTDemucs from 5.6 GB to
    /// 2.1 GB at an 11% throughput cost (docs/parity/2026-09-24-memory.md):
    /// constant folding off (it materializes about 3.8 GB of index tensors
    /// at load) and no memory-pattern pre-planning.
    pub fn low_memory() -> Self {
        Self {
            memory_pattern: false,
            disabled_optimizers: vec!["ConstantFolding".to_string()],
            ..Self::default()
        }
    }

    /// ONNX Runtime defaults: fastest measured, highest memory
    pub fn max_speed() -> Self {
        Self::default()
    }
}

/// Input length of the HTDemucs ONNX export: 7.8 s at 44.1 kHz.
pub const HTDEMUCS_SEGMENT_SAMPLES: usize = 343_980;

impl ModelConfig {
    /// Configuration for the 4-stem HTDemucs ONNX export
    /// (`StemSplitio/htdemucs-onnx`, `htdemucs.onnx`): input `mix`
    /// `[1, 2, 343980]`, output `stems` `[1, 4, 2, 343980]` in the order
    /// drums, bass, other, vocals.
    pub fn htdemucs<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            backend: None,
            sample_rate: 44100,
            channels: 2,
            sources: ["drums", "bass", "other", "vocals"]
                .map(String::from)
                .to_vec(),
            chunk_size: None,
            segment_samples: Some(HTDEMUCS_SEGMENT_SAMPLES),
            input_name: default_input_name(),
            output_name: default_output_name(),
            onnx: OnnxOptions::low_memory(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("model.onnx"),
            backend: None, // Will be inferred from file extension
            sample_rate: 44100,
            channels: 2,
            sources: vec![
                "drums".to_string(),
                "bass".to_string(),
                "vocals".to_string(),
                "other".to_string(),
            ],
            chunk_size: Some(441000), // 10 seconds at 44.1kHz
            segment_samples: None,
            input_name: default_input_name(),
            output_name: default_output_name(),
            onnx: OnnxOptions::default(),
        }
    }
}

/// ONNX Runtime model wrapper
#[cfg(feature = "ort-backend")]
pub struct OnnxModel {
    // `Session::run` needs `&mut self`; ONNX Runtime parallelizes inside a
    // single run, so segments are processed one at a time.
    session: Mutex<Session>,
    config: ModelConfig,
}

#[cfg(feature = "ort-backend")]
impl OnnxModel {
    /// Create new ONNX model
    pub fn new(config: ModelConfig) -> Result<Self> {
        let model_err = |e: ort::Error<_>| CharonError::Model(e.to_string());
        let opts = &config.onnx;
        let threads = opts.intra_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        let level = match opts.optimization_level {
            OptimizationLevel::Disable => GraphOptimizationLevel::Disable,
            OptimizationLevel::Basic => GraphOptimizationLevel::Level1,
            OptimizationLevel::Extended => GraphOptimizationLevel::Level2,
            OptimizationLevel::All => GraphOptimizationLevel::Level3,
        };
        let mut builder = Session::builder()?
            .with_optimization_level(level)
            .map_err(model_err)?
            .with_intra_threads(threads)
            .map_err(model_err)?
            .with_memory_pattern(opts.memory_pattern)
            .map_err(model_err)?;
        if !opts.disabled_optimizers.is_empty() {
            builder = builder
                .with_disabled_optimizers(opts.disabled_optimizers.join(";"))
                .map_err(model_err)?;
        }
        // The CPU EP is always registered last as the fallback; its arena
        // setting is what `cpu_arena` controls.
        let cpu = ort::ep::CPU::default()
            .with_arena_allocator(opts.cpu_arena)
            .build();
        builder = match opts.execution_provider {
            ExecutionProvider::Cpu => builder.with_execution_providers([cpu]).map_err(model_err)?,
            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreMl => {
                // MLProgram is the current CoreML format; the model's time axis
                // is static, which lets CoreML compile fixed shapes. The
                // compiled model is cached next to the ONNX file.
                let cache_dir = config
                    .model_path
                    .parent()
                    .map(|p| p.join("coreml-cache"))
                    .unwrap_or_else(|| std::path::PathBuf::from("coreml-cache"));
                std::fs::create_dir_all(&cache_dir)?;
                builder
                    .with_execution_providers([
                        ort::ep::CoreML::default()
                            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                            .with_static_input_shapes(true)
                            .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                            .with_model_cache_dir(cache_dir.display())
                            .build(),
                        cpu,
                    ])
                    .map_err(model_err)?
            }
            #[cfg(not(feature = "coreml"))]
            ExecutionProvider::CoreMl => {
                return Err(CharonError::NotSupported(
                    "CoreML execution provider needs the `coreml` feature".to_string(),
                ))
            }
        };
        let session = builder.commit_from_file(&config.model_path)?;

        Ok(Self {
            session: Mutex::new(session),
            config,
        })
    }

    /// Run inference on one segment of audio, shaped `[channels, samples]`.
    ///
    /// Returns one `[channels, samples]` array per source, in the order of
    /// `ModelConfig::sources`.
    pub fn infer(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        let (channels, samples) = input.dim();
        if let Some(fixed) = self.config.segment_samples {
            if samples != fixed {
                return Err(CharonError::Model(format!(
                    "model expects {fixed} samples per segment, got {samples}"
                )));
            }
        }

        let data: Vec<f32> = input.iter().copied().collect();
        let tensor = Tensor::from_array(([1, channels, samples], data))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| CharonError::Model("ONNX session lock poisoned".to_string()))?;
        let outputs = session.run(ort::inputs![self.config.input_name.as_str() => tensor])?;
        let output = outputs.get(&self.config.output_name).ok_or_else(|| {
            CharonError::Model(format!(
                "model has no output named '{}'",
                self.config.output_name
            ))
        })?;
        let (shape, values) = output.try_extract_tensor::<f32>()?;

        let num_sources = self.config.sources.len();
        let expected = [1, num_sources as i64, channels as i64, samples as i64];
        if shape[..] != expected[..] {
            return Err(CharonError::Model(format!(
                "model output shape {:?}, expected {expected:?}",
                &shape[..]
            )));
        }

        let per_source = channels * samples;
        values
            .chunks_exact(per_source)
            .map(|source| {
                Array2::from_shape_vec((channels, samples), source.to_vec())
                    .map_err(|e| CharonError::Model(e.to_string()))
            })
            .collect()
    }
}

/// Generic model interface
pub enum Model {
    #[cfg(feature = "ort-backend")]
    Onnx(OnnxModel),
}

impl Model {
    /// Create a new model from configuration
    pub fn from_config(config: ModelConfig) -> Result<Self> {
        #[cfg(feature = "ort-backend")]
        {
            let is_onnx = config.backend == Some(ModelBackend::OnnxRuntime)
                || config
                    .model_path
                    .extension()
                    .is_some_and(|ext| ext == "onnx");
            if is_onnx {
                return Ok(Model::Onnx(OnnxModel::new(config)?));
            }
        }
        #[cfg(not(feature = "ort-backend"))]
        let _ = config;

        Err(CharonError::NotSupported(
            "No ML backend enabled or auto-detected".to_string(),
        ))
    }

    /// Run inference
    pub fn infer(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        #[cfg(not(feature = "ort-backend"))]
        let _ = input;
        match *self {
            #[cfg(feature = "ort-backend")]
            Model::Onnx(ref model) => model.infer(input),
        }
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        match *self {
            #[cfg(feature = "ort-backend")]
            Model::Onnx(ref model) => &model.config,
        }
    }
}

/// Model registry for managing pre-trained models
pub struct ModelRegistry {
    models_dir: PathBuf,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
        }
    }

    /// List available models
    pub fn list_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();

        if !self.models_dir.exists() {
            return Ok(models);
        }

        for entry in std::fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "onnx" || ext == "safetensors" {
                        if let Some(name) = path.file_stem() {
                            models.push(name.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        Ok(models)
    }

    /// Get model path by name
    pub fn get_model_path(&self, name: &str) -> Option<PathBuf> {
        let onnx_path = self.models_dir.join(format!("{name}.onnx"));
        if onnx_path.exists() {
            return Some(onnx_path);
        }

        let safetensors_path = self.models_dir.join(format!("{name}.safetensors"));
        if safetensors_path.exists() {
            return Some(safetensors_path);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.sources.len(), 4);
    }
}

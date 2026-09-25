//! ML model backends and configuration

use crate::error::{CharonError, Result};
#[cfg(feature = "ort-backend")]
use crate::stft::{DemucsStft, Spectrogram};
use ndarray::Array2;
#[cfg(feature = "ort-backend")]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
#[cfg(feature = "ort-backend")]
use rayon::prelude::*;
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
    /// Tensor contract
    #[serde(default)]
    pub contract: ModelContract,
    /// ONNX Runtime session options
    #[serde(default)]
    pub onnx: OnnxOptions,
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
    Cpu,
    /// Apple CoreML (GPU), with CPU fallback for unsupported operators.
    /// Needs the `coreml` feature. Runs the split-transform HTDemucs
    /// export; the in-graph-STFT export fails on it
    /// (docs/MEASUREMENTS.md).
    CoreMl,
    /// CoreML when the feature is enabled and the session builds, else CPU
    #[default]
    Auto,
}

/// How a model's tensors map onto audio
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelContract {
    /// One waveform in (`[1, channels, samples]`), stems out
    /// (`[1, sources, channels, samples]`).
    Waveform { input: String, output: String },
    /// Demucs with the transforms outside the graph: waveform `mix`
    /// `[1, C, L]` and complex-as-channels spectrogram `spec`
    /// `[1, 2C, F, T]` in; time branch `time` `[1, S, C, L]` and
    /// spectral branch `spec_out` `[1, S, 2C, F, T]` out. The host runs
    /// `HTDemucs._spec` / `_ispec` and returns `time + ispec(spec_out)`.
    DemucsSplit {
        mix: String,
        spec: String,
        time: String,
        spec_out: String,
    },
}

impl Default for ModelContract {
    fn default() -> Self {
        ModelContract::Waveform {
            input: "mix".to_string(),
            output: "stems".to_string(),
        }
    }
}

impl ModelContract {
    /// The split contract with the tensor names of `tools/export/export_htdemucs.py`
    pub fn demucs_split() -> Self {
        ModelContract::DemucsSplit {
            mix: "mix".to_string(),
            spec: "spec".to_string(),
            time: "time".to_string(),
            spec_out: "spec_out".to_string(),
        }
    }
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
    /// Directory for the compiled CoreML model. Compiling takes 25-40 s;
    /// a cached model loads in a few seconds. Default: `coreml-cache`
    /// next to the model file.
    pub coreml_cache_dir: Option<PathBuf>,
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
            execution_provider: ExecutionProvider::Auto,
            coreml_cache_dir: None,
        }
    }
}

impl OnnxOptions {
    /// Settings measured to cut peak memory on HTDemucs from 5.6 GB to
    /// 2.1 GB at an 11% throughput cost (docs/MEASUREMENTS.md):
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
            contract: ModelContract::default(),
            onnx: OnnxOptions {
                // The in-graph STFT export only runs on the CPU provider.
                execution_provider: ExecutionProvider::Cpu,
                ..OnnxOptions::low_memory()
            },
        }
    }

    /// Configuration for the split-transform HTDemucs export produced by
    /// `tools/export/export_htdemucs.py`: STFT/iSTFT run in charon, the
    /// network runs on CoreML when available, else CPU.
    pub fn htdemucs_split<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            contract: ModelContract::demucs_split(),
            // No index constants to fold any more; memory-pattern planning
            // still costs 2 GB of peak on the CPU path for 6% of time
            // (docs/MEASUREMENTS.md).
            onnx: OnnxOptions {
                memory_pattern: false,
                // Level 3 adds layout transforms that cost 4% on this graph
                // on Apple Silicon (docs/MEASUREMENTS.md).
                optimization_level: OptimizationLevel::Extended,
                ..OnnxOptions::default()
            },
            ..Self::htdemucs(model_path)
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
            contract: ModelContract::default(),
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
    stft: Option<DemucsStft>,
    provider: &'static str,
}

#[cfg(feature = "ort-backend")]
impl OnnxModel {
    /// Create new ONNX model
    pub fn new(config: ModelConfig) -> Result<Self> {
        let (session, provider) = match config.onnx.execution_provider {
            ExecutionProvider::Cpu => (Self::build_session(&config, false)?, "CPU"),
            ExecutionProvider::CoreMl => (Self::build_session(&config, true)?, "CoreML"),
            ExecutionProvider::Auto => match Self::build_session(&config, true) {
                Ok(session) if cfg!(feature = "coreml") => (session, "CoreML"),
                Ok(session) => (session, "CPU"),
                Err(e) => {
                    log::warn!("CoreML session failed ({e}); falling back to CPU");
                    (Self::build_session(&config, false)?, "CPU")
                }
            },
        };
        log::info!("ONNX session on {provider}");
        let stft = match config.contract {
            ModelContract::Waveform { .. } => None,
            ModelContract::DemucsSplit { .. } => Some(DemucsStft::htdemucs()),
        };
        Ok(Self {
            session: Mutex::new(session),
            config,
            stft,
            provider,
        })
    }

    /// Execution provider the session runs on: "CPU" or "CoreML"
    pub fn provider(&self) -> &'static str {
        self.provider
    }

    fn build_session(config: &ModelConfig, coreml: bool) -> Result<Session> {
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
        builder = if coreml {
            Self::with_coreml(builder, config, cpu)?
        } else {
            builder.with_execution_providers([cpu]).map_err(model_err)?
        };
        Ok(builder.commit_from_file(&config.model_path)?)
    }

    #[cfg(feature = "coreml")]
    fn with_coreml(
        builder: ort::session::builder::SessionBuilder,
        config: &ModelConfig,
        cpu: ort::ep::ExecutionProviderDispatch,
    ) -> Result<ort::session::builder::SessionBuilder> {
        // MLProgram is the current CoreML format; the model's time axis is
        // static, which lets CoreML compile fixed shapes.
        // ONNX Runtime keys the compiled model by path, not content: a
        // re-exported model with the same name would load a stale compiled
        // model and fail at run time. Key the directory by the file's hash.
        let base = config.onnx.coreml_cache_dir.clone().unwrap_or_else(|| {
            config
                .model_path
                .parent()
                .map(|p| p.join("coreml-cache"))
                .unwrap_or_else(|| PathBuf::from("coreml-cache"))
        });
        let cache_dir = base.join(&file_sha256_prefix(&config.model_path)?);
        std::fs::create_dir_all(&cache_dir)?;
        builder
            .with_execution_providers([
                ort::ep::CoreML::default()
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .with_static_input_shapes(true)
                    .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndGPU)
                    .with_model_cache_dir(cache_dir.display())
                    .build(),
                cpu,
            ])
            .map_err(|e| CharonError::Model(e.to_string()))
    }

    #[cfg(not(feature = "coreml"))]
    fn with_coreml(
        builder: ort::session::builder::SessionBuilder,
        config: &ModelConfig,
        cpu: ort::ep::ExecutionProviderDispatch,
    ) -> Result<ort::session::builder::SessionBuilder> {
        if config.onnx.execution_provider == ExecutionProvider::CoreMl {
            return Err(CharonError::NotSupported(
                "CoreML execution provider needs the `coreml` feature".to_string(),
            ));
        }
        builder
            .with_execution_providers([cpu])
            .map_err(|e| CharonError::Model(e.to_string()))
    }

    /// Run inference on one segment of audio, shaped `[channels, samples]`.
    ///
    /// Returns one `[channels, samples]` array per source, in the order of
    /// `ModelConfig::sources`.
    pub fn infer(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        let samples = input.ncols();
        if let Some(fixed) = self.config.segment_samples {
            if samples != fixed {
                return Err(CharonError::Model(format!(
                    "model expects {fixed} samples per segment, got {samples}"
                )));
            }
        }
        match &self.config.contract {
            ModelContract::Waveform {
                input: name,
                output,
            } => self.infer_waveform(input, name, output),
            ModelContract::DemucsSplit {
                mix,
                spec,
                time,
                spec_out,
            } => self.infer_split(input, mix, spec, time, spec_out),
        }
    }

    fn infer_waveform(
        &self,
        input: &Array2<f32>,
        input_name: &str,
        output_name: &str,
    ) -> Result<Vec<Array2<f32>>> {
        let (channels, samples) = input.dim();
        let data: Vec<f32> = input.iter().copied().collect();
        let tensor = Tensor::from_array(([1, channels, samples], data))?;

        let mut session = self.lock_session()?;
        let outputs = session.run(ort::inputs![input_name => tensor])?;
        let output = outputs.get(output_name).ok_or_else(|| {
            CharonError::Model(format!("model has no output named '{output_name}'"))
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

    fn infer_split(
        &self,
        input: &Array2<f32>,
        mix_name: &str,
        spec_name: &str,
        time_name: &str,
        spec_out_name: &str,
    ) -> Result<Vec<Array2<f32>>> {
        let stft = self.stft.as_ref().expect("split contract has an STFT");
        let (channels, samples) = input.dim();
        let (freqs, frames) = (stft.freqs(), stft.frames(samples));

        // Spectrogram of each channel, laid out [1, 2C, F, T] with re/im
        // planes per channel (HTDemucs._magnitude with cac=True).
        let specs: Vec<Spectrogram> = (0..channels)
            .into_par_iter()
            .map(|ch| {
                let row = input.row(ch);
                stft.spec(row.as_slice().expect("contiguous row"))
            })
            .collect::<Result<_>>()?;
        let mut spec_data = vec![0.0f32; 2 * channels * freqs * frames];
        for (ch, spec) in specs.iter().enumerate() {
            let re = &mut spec_data[(2 * ch) * freqs * frames..(2 * ch + 1) * freqs * frames];
            for t in 0..frames {
                for f in 0..freqs {
                    re[f * frames + t] = spec.data[t * freqs + f].re;
                }
            }
            let im = &mut spec_data[(2 * ch + 1) * freqs * frames..(2 * ch + 2) * freqs * frames];
            for t in 0..frames {
                for f in 0..freqs {
                    im[f * frames + t] = spec.data[t * freqs + f].im;
                }
            }
        }
        let mix_data: Vec<f32> = input.iter().copied().collect();
        let mix_tensor = Tensor::from_array(([1, channels, samples], mix_data))?;
        let spec_tensor = Tensor::from_array(([1, 2 * channels, freqs, frames], spec_data))?;

        let num_sources = self.config.sources.len();
        let (time, spec_out) = {
            let mut session = self.lock_session()?;
            let outputs =
                session.run(ort::inputs![mix_name => mix_tensor, spec_name => spec_tensor])?;
            let time = outputs
                .get(time_name)
                .ok_or_else(|| CharonError::Model(format!("model has no output '{time_name}'")))?;
            let (shape, values) = time.try_extract_tensor::<f32>()?;
            let expected = [1, num_sources as i64, channels as i64, samples as i64];
            if shape[..] != expected[..] {
                return Err(CharonError::Model(format!(
                    "time output shape {:?}, expected {expected:?}",
                    &shape[..]
                )));
            }
            let time = values.to_vec();
            let spec_out = outputs.get(spec_out_name).ok_or_else(|| {
                CharonError::Model(format!("model has no output '{spec_out_name}'"))
            })?;
            let (shape, values) = spec_out.try_extract_tensor::<f32>()?;
            let expected = [
                1,
                num_sources as i64,
                2 * channels as i64,
                freqs as i64,
                frames as i64,
            ];
            if shape[..] != expected[..] {
                return Err(CharonError::Model(format!(
                    "spectral output shape {:?}, expected {expected:?}",
                    &shape[..]
                )));
            }
            (time, values.to_vec())
        };

        // stems = time + ispec(spec_out), one iSTFT per (source, channel)
        let plane = freqs * frames;
        let waves: Vec<Vec<f32>> = (0..num_sources * channels)
            .into_par_iter()
            .map(|idx| {
                let (s, ch) = (idx / channels, idx % channels);
                let base = (s * 2 * channels + 2 * ch) * plane;
                let re = &spec_out[base..base + plane];
                let im = &spec_out[base + plane..base + 2 * plane];
                let mut data = Vec::with_capacity(plane);
                for t in 0..frames {
                    for f in 0..freqs {
                        data.push(realfft::num_complex::Complex::new(
                            re[f * frames + t],
                            im[f * frames + t],
                        ));
                    }
                }
                let spec = Spectrogram {
                    freqs,
                    frames,
                    data,
                };
                let mut wave = stft.ispec(&spec, samples)?;
                let t = &time[(s * channels + ch) * samples..(s * channels + ch + 1) * samples];
                for (w, &x) in wave.iter_mut().zip(t) {
                    *w += x;
                }
                Ok(wave)
            })
            .collect::<Result<_>>()?;

        (0..num_sources)
            .map(|s| {
                let mut data = Vec::with_capacity(channels * samples);
                for ch in 0..channels {
                    data.extend_from_slice(&waves[s * channels + ch]);
                }
                Array2::from_shape_vec((channels, samples), data)
                    .map_err(|e| CharonError::Model(e.to_string()))
            })
            .collect()
    }

    fn lock_session(&self) -> Result<std::sync::MutexGuard<'_, Session>> {
        self.session
            .lock()
            .map_err(|_| CharonError::Model("ONNX session lock poisoned".to_string()))
    }
}

/// First 16 hex digits of the SHA-256 of a file (about 0.2 s for 200 MB)
#[cfg(feature = "coreml")]
fn file_sha256_prefix(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize())[..16].to_string())
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

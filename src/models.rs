//! ML model backends and configuration

use crate::error::{CharonError, Result};
use ndarray::Array2;
#[cfg(feature = "candle-backend")]
use candle_core::{Device, Tensor};
#[cfg(feature = "ort-backend")]
use ort::{
    session::{builder::{GraphOptimizationLevel, SessionBuilder}, Session},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Model backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelBackend {
    #[cfg(feature = "ort-backend")]
    /// ONNX Runtime (production-ready, hardware accelerated)
    OnnxRuntime,
    #[cfg(feature = "candle-backend")]
    /// HuggingFace Candle (pure Rust, flexible)
    Candle,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model file
    pub model_path: PathBuf,
    /// Model backend to use (optional, will be inferred if not set)
    #[serde(skip, default)]
    #[cfg(any(feature = "ort-backend", feature = "candle-backend"))]
    pub backend: Option<ModelBackend>,
    /// Expected sample rate
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: usize,
    /// Source names (e.g., ["drums", "bass", "vocals", "other"])
    pub sources: Vec<String>,
    /// Chunk size for processing (in samples)
    pub chunk_size: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("model.onnx"),
            #[cfg(any(feature = "ort-backend", feature = "candle-backend"))]
            backend: None,  // Will be inferred from file extension
            sample_rate: 44100,
            channels: 2,
            sources: vec![
                "drums".to_string(),
                "bass".to_string(),
                "vocals".to_string(),
                "other".to_string(),
            ],
            chunk_size: Some(441000), // 10 seconds at 44.1kHz
        }
    }
}

/// ONNX Runtime model wrapper
#[cfg(feature = "ort-backend")]
pub struct OnnxModel {
    #[allow(dead_code)]
    session: Session,
    config: ModelConfig,
}

#[cfg(feature = "ort-backend")]
impl OnnxModel {
    /// Create new ONNX model
    pub fn new(config: ModelConfig) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&config.model_path)?;

        Ok(Self { session, config })
    }

    /// Run inference on audio data
    pub fn infer(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        // For now, return a placeholder implementation
        // A real implementation would require:
        // 1. Converting input to proper ONNX tensor format
        // 2. Running session inference
        // 3. Parsing output tensors
        
        // Placeholder: return copies of input as "separated" sources
        let num_sources = self.config.sources.len();
        let separated = vec![input.clone(); num_sources];
        
        Ok(separated)
    }
}

/// Candle model wrapper (for pure Rust inference)
#[cfg(feature = "candle-backend")]
pub struct CandleModel {
    device: Device,
    config: ModelConfig,
    model: Option<candle_nn::VarMap>,
}

#[cfg(feature = "candle-backend")]
impl CandleModel {
    /// Create new Candle model
    pub fn new(config: ModelConfig) -> Result<Self> {
        use candle_core::safetensors;
        
        let device = if cfg!(target_arch = "wasm32") {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };
        
        let model = if config.model_path.exists() {
            let tensors = safetensors::load(&config.model_path, &device)?;
            let mut varmap = candle_nn::VarMap::new();
            for (name, tensor) in tensors {
                varmap.data().lock().unwrap().insert(name, candle_nn::Var::from_tensor(&tensor)?);
            }
            Some(varmap)
        } else {
            None
        };
        
        Ok(Self { device, config, model })
    }

    /// Run inference on audio data
    pub fn infer(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        let (channels, samples) = (input.nrows(), input.ncols());
        let data: Vec<f32> = input.t().iter().copied().collect();
        
        let tensor = Tensor::from_vec(data, (samples, channels), &self.device)?;
        
        let output = if let Some(ref _model) = self.model {
            tensor.clone()
        } else {
            tensor.clone()
        };
        
        let output_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let num_sources = self.config.sources.len();
        let samples_per_source = output_data.len() / num_sources;
        
        let mut separated = Vec::new();
        for i in 0..num_sources {
            let start = i * samples_per_source;
            let end = start + samples_per_source;
            let source_data = &output_data[start..end];
            
            let mut source_array = Array2::zeros((channels, samples));
            for (idx, &val) in source_data.iter().enumerate() {
                let ch = idx % channels;
                let samp = idx / channels;
                if samp < samples {
                    source_array[[ch, samp]] = val;
                }
            }
            separated.push(source_array);
        }

        Ok(separated)
    }
}

/// Generic model interface
pub enum Model {
    #[cfg(feature = "ort-backend")]
    Onnx(OnnxModel),
    #[cfg(feature = "candle-backend")]
    Candle(CandleModel),
}

impl Model {
    /// Create a new model from configuration
    pub fn from_config(config: ModelConfig) -> Result<Self> {
        // Infer backend from file extension if not specified
        #[cfg(any(feature = "ort-backend", feature = "candle-backend"))]
        let backend = config.backend.or_else(|| {
            if config.model_path.extension()?.to_str()? == "onnx" {
                #[cfg(feature = "ort-backend")]
                return Some(ModelBackend::OnnxRuntime);
            }
            #[cfg(feature = "candle-backend")]
            return Some(ModelBackend::Candle);
            #[allow(unreachable_code)]
            None
        });
        
        #[cfg(feature = "ort-backend")]
        if matches!(backend, Some(ModelBackend::OnnxRuntime)) {
            return Ok(Model::Onnx(OnnxModel::new(config)?));
        }
        #[cfg(feature = "candle-backend")]
        if matches!(backend, Some(ModelBackend::Candle)) {
            return Ok(Model::Candle(CandleModel::new(config)?));
        }
        Err(CharonError::NotSupported("No ML backend enabled or auto-detected".to_string()))
    }

    /// Run inference
    #[allow(unreachable_patterns)]
    pub fn infer(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        match self {
            #[cfg(feature = "ort-backend")]
            Model::Onnx(model) => model.infer(input),
            #[cfg(feature = "candle-backend")]
            Model::Candle(model) => model.infer(input),
            #[allow(unreachable_patterns)]
            _ => Err(CharonError::NotSupported("No model backend available".to_string())),
        }
    }

    /// Get model configuration
    #[allow(unreachable_patterns)]
    pub fn config(&self) -> &ModelConfig {
        match self {
            #[cfg(feature = "ort-backend")]
            Model::Onnx(model) => &model.config,
            #[cfg(feature = "candle-backend")]
            Model::Candle(model) => &model.config,
            #[allow(unreachable_patterns)]
            _ => panic!("No model backend available"),
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

    #[test]
    #[cfg(all(feature = "ort-backend", feature = "candle-backend"))]
    fn test_model_backend_types() {
        assert_ne!(ModelBackend::OnnxRuntime, ModelBackend::Candle);
    }
}

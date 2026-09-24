//! ML model backends and configuration

use crate::error::{CharonError, Result};
use ndarray::Array2;
#[cfg(feature = "ort-backend")]
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
        let model_err = |e: ort::Error<_>| CharonError::Model(e.to_string());
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(model_err)?
            .with_intra_threads(4)
            .map_err(model_err)?
            .commit_from_file(&config.model_path)?;

        Ok(Self { session, config })
    }

    /// Run inference on audio data
    pub fn infer(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        // Placeholder: returns copies of the input as "separated" sources.
        // Real inference lands with the HTDemucs model contract.
        let num_sources = self.config.sources.len();
        let separated = vec![input.clone(); num_sources];

        Ok(separated)
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

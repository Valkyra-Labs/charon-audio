//! Model metadata registry.
//!
//! Holds metadata (source URL, hash, stems) for known models and locates
//! downloaded files. It does not download: `download_model` returns the
//! URL to fetch manually.

use crate::error::{CharonError, Result};
use crate::models::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub sources: Vec<String>,
    pub sample_rate: u32,
    pub channels: usize,
    pub file_size_mb: f64,
    pub download_url: Option<String>,
    /// SHA-256 of the model file, when known
    #[serde(default)]
    pub sha256: Option<String>,
}

/// Pre-trained model zoo
pub struct ModelZoo {
    models_dir: PathBuf,
    registry: HashMap<String, ModelMetadata>,
}

impl ModelZoo {
    /// Create new model zoo
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Result<Self> {
        let models_dir = models_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&models_dir)?;

        let mut zoo = Self {
            models_dir,
            registry: HashMap::new(),
        };

        zoo.register_builtin_models();
        Ok(zoo)
    }

    /// Register built-in models
    fn register_builtin_models(&mut self) {
        self.registry.insert(
            "htdemucs".to_string(),
            ModelMetadata {
                name: "htdemucs".to_string(),
                version: "StemSplitio/htdemucs-onnx".to_string(),
                description: "HTDemucs 4-stem ONNX export (drums, bass, other, vocals). \
                              Weights license not stated by the model authors."
                    .to_string(),
                sources: ["drums", "bass", "other", "vocals"]
                    .map(String::from)
                    .to_vec(),
                sample_rate: 44100,
                channels: 2,
                file_size_mb: 301.8,
                download_url: Some(
                    "https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx"
                        .to_string(),
                ),
                sha256: Some(
                    "68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74".to_string(),
                ),
            },
        );
    }

    /// List available models
    pub fn list_models(&self) -> Vec<&ModelMetadata> {
        self.registry.values().collect()
    }

    /// Get model metadata by name
    pub fn get_metadata(&self, name: &str) -> Option<&ModelMetadata> {
        self.registry.get(name)
    }

    /// Check if model is downloaded
    pub fn is_downloaded(&self, name: &str) -> bool {
        self.get_model_path(name).is_some_and(|p| p.exists())
    }

    /// Get model path
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

    /// Downloading is not implemented; returns the path if the file is
    /// already present, otherwise an error naming the download URL.
    pub fn download_model(&self, name: &str) -> Result<PathBuf> {
        let metadata = self
            .get_metadata(name)
            .ok_or_else(|| CharonError::NotSupported(format!("Model {name} not found")))?;

        let download_url = metadata
            .download_url
            .as_ref()
            .ok_or_else(|| CharonError::NotSupported("No download URL available".to_string()))?;

        let target_path = self.models_dir.join(format!("{name}.onnx"));

        if target_path.exists() {
            return Ok(target_path);
        }

        Err(CharonError::NotSupported(format!(
            "Model download not implemented. Please manually download from: {download_url}"
        )))
    }

    /// Load model configuration
    pub fn load_model(&self, name: &str) -> Result<ModelConfig> {
        let metadata = self
            .get_metadata(name)
            .ok_or_else(|| CharonError::NotSupported(format!("Model {name} not found")))?;

        let model_path = self
            .get_model_path(name)
            .ok_or_else(|| CharonError::NotSupported(format!("Model {name} not downloaded")))?;

        // The only registered contract is HTDemucs; other entries would
        // need their own tensor names and segment length.
        let mut config = ModelConfig::htdemucs(&model_path);
        config.sample_rate = metadata.sample_rate;
        config.channels = metadata.channels;
        config.sources = metadata.sources.clone();
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_zoo_creation() {
        let temp_dir = std::env::temp_dir().join("charon_test_zoo");
        let zoo = ModelZoo::new(&temp_dir).unwrap();
        assert!(!zoo.list_models().is_empty());
    }

    #[test]
    fn test_model_metadata() {
        let temp_dir = std::env::temp_dir().join("charon_test_zoo");
        let zoo = ModelZoo::new(&temp_dir).unwrap();
        let metadata = zoo.get_metadata("htdemucs").unwrap();
        assert_eq!(metadata.sources.len(), 4);
        assert!(metadata.sha256.is_some());
    }
}

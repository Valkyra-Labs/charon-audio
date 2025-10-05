//! Pre-trained model zoo for easy model access

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
            "demucs-4stems".to_string(),
            ModelMetadata {
                name: "demucs-4stems".to_string(),
                version: "1.0.0".to_string(),
                description: "Demucs 4-stem separation (drums, bass, vocals, other)".to_string(),
                sources: vec!["drums".to_string(), "bass".to_string(), "vocals".to_string(), "other".to_string()],
                sample_rate: 44100,
                channels: 2,
                file_size_mb: 150.0,
                download_url: Some("https://example.com/models/demucs-4stems.onnx".to_string()),
            },
        );

        self.registry.insert(
            "demucs-6stems".to_string(),
            ModelMetadata {
                name: "demucs-6stems".to_string(),
                version: "1.0.0".to_string(),
                description: "Demucs 6-stem separation (drums, bass, vocals, other, piano, guitar)".to_string(),
                sources: vec!["drums".to_string(), "bass".to_string(), "vocals".to_string(), "other".to_string(), "piano".to_string(), "guitar".to_string()],
                sample_rate: 44100,
                channels: 2,
                file_size_mb: 200.0,
                download_url: Some("https://example.com/models/demucs-6stems.onnx".to_string()),
            },
        );

        self.registry.insert(
            "vocals-only".to_string(),
            ModelMetadata {
                name: "vocals-only".to_string(),
                version: "1.0.0".to_string(),
                description: "Optimized vocal extraction model".to_string(),
                sources: vec!["vocals".to_string(), "instrumental".to_string()],
                sample_rate: 44100,
                channels: 2,
                file_size_mb: 80.0,
                download_url: Some("https://example.com/models/vocals-only.onnx".to_string()),
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

    /// Download model (placeholder - requires actual HTTP client)
    pub fn download_model(&self, name: &str) -> Result<PathBuf> {
        let metadata = self.get_metadata(name)
            .ok_or_else(|| CharonError::NotSupported(format!("Model {name} not found")))?;

        let download_url = metadata.download_url.as_ref()
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
        let metadata = self.get_metadata(name)
            .ok_or_else(|| CharonError::NotSupported(format!("Model {name} not found")))?;

        let model_path = self.get_model_path(name)
            .ok_or_else(|| CharonError::NotSupported(format!("Model {name} not downloaded")))?;

        Ok(ModelConfig {
            model_path,
            #[cfg(any(feature = "ort-backend", feature = "candle-backend"))]
            backend: None,
            sample_rate: metadata.sample_rate,
            channels: metadata.channels,
            sources: metadata.sources.clone(),
            chunk_size: Some(441000),
        })
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
        let metadata = zoo.get_metadata("demucs-4stems");
        assert!(metadata.is_some());
        assert_eq!(metadata.unwrap().sources.len(), 4);
    }
}

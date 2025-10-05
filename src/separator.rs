//! Main separator API

use crate::audio::{AudioBuffer, AudioFile};
use crate::error::{CharonError, Result};
use crate::models::{Model, ModelBackend, ModelConfig};
use crate::processor::{ProcessConfig, Processor};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Separator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparatorConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Processing configuration
    pub process: ProcessConfig,
    /// Show progress bars
    pub show_progress: bool,
}

impl Default for SeparatorConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            process: ProcessConfig::default(),
            show_progress: true,
        }
    }
}

impl SeparatorConfig {
    /// Create configuration for ONNX backend
    #[cfg(feature = "ort-backend")]
    pub fn onnx<P: AsRef<Path>>(model_path: P) -> Self {
        let mut config = Self::default();
        config.model.model_path = model_path.as_ref().to_path_buf();
        config.model.backend = Some(ModelBackend::OnnxRuntime);
        config
    }

    /// Create configuration for Candle backend
    #[cfg(feature = "candle-backend")]
    pub fn candle<P: AsRef<Path>>(model_path: P) -> Self {
        let mut config = Self::default();
        config.model.model_path = model_path.as_ref().to_path_buf();
        config.model.backend = Some(ModelBackend::Candle);
        config
    }

    /// Set number of ensemble shifts
    pub fn with_shifts(mut self, shifts: usize) -> Self {
        self.process.shifts = shifts;
        self
    }

    /// Set segment length
    pub fn with_segment_length(mut self, seconds: f64) -> Self {
        self.process.segment_length = Some(seconds);
        self
    }

    /// Enable/disable progress display
    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }
}

/// Separated audio stems
pub struct Stems {
    /// Map of source name to audio buffer
    pub sources: HashMap<String, AudioBuffer>,
}

impl Stems {
    /// Create new stems collection
    pub fn new(sources: HashMap<String, AudioBuffer>) -> Self {
        Self { sources }
    }

    /// Get stem by name
    pub fn get(&self, name: &str) -> Option<&AudioBuffer> {
        self.sources.get(name)
    }

    /// Save all stems to directory
    pub fn save_all<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        for (name, buffer) in &self.sources {
            let output_path = output_dir.join(format!("{name}.wav"));
            AudioFile::write_wav(&output_path, buffer)?;
        }

        Ok(())
    }

    /// Save specific stem
    pub fn save<P: AsRef<Path>>(&self, name: &str, path: P) -> Result<()> {
        let buffer = self
            .sources
            .get(name)
            .ok_or_else(|| CharonError::Audio(format!("Stem '{name}' not found")))?;
        AudioFile::write_wav(path, buffer)
    }

    /// List available stem names
    pub fn list(&self) -> Vec<String> {
        self.sources.keys().cloned().collect()
    }
}

/// Main separator for audio source separation
pub struct Separator {
    model: Model,
    processor: Processor,
    config: SeparatorConfig,
}

impl Separator {
    /// Create new separator from configuration
    pub fn new(config: SeparatorConfig) -> Result<Self> {
        let model = Model::from_config(config.model.clone())?;
        let processor = Processor::new(config.process.clone());

        Ok(Self {
            model,
            processor,
            config,
        })
    }

    /// Create separator with default configuration
    pub fn with_default_model() -> Result<Self> {
        Self::new(SeparatorConfig::default())
    }

    /// Separate audio buffer into stems
    pub fn separate(&self, audio: &AudioBuffer) -> Result<Stems> {
        // Resample if needed
        let audio = if audio.sample_rate != self.config.model.sample_rate {
            audio.resample(self.config.model.sample_rate)?
        } else {
            audio.clone()
        };

        // Convert channels if needed
        let audio = if audio.channels() != self.config.model.channels {
            audio.convert_channels(self.config.model.channels)?
        } else {
            audio
        };

        // Create progress bar
        let pb = if self.config.show_progress {
            let pb = ProgressBar::new(100);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            pb.set_message("Separating audio...");
            Some(pb)
        } else {
            None
        };

        // Process audio
        let separated = self.processor.process(&self.model, &audio)?;

        if let Some(pb) = &pb {
            pb.finish_with_message("Separation complete!");
        }

        // Build stems map
        let mut sources = HashMap::new();
        for (idx, buffer) in separated.into_iter().enumerate() {
            if idx < self.config.model.sources.len() {
                let name = &self.config.model.sources[idx];
                sources.insert(name.clone(), buffer);
            }
        }

        Ok(Stems::new(sources))
    }

    /// Separate audio from file
    pub fn separate_file<P: AsRef<Path>>(&self, path: P) -> Result<Stems> {
        let audio = AudioFile::read(path)?;
        self.separate(&audio)
    }

    /// Separate audio and save stems
    pub fn separate_and_save<P: AsRef<Path>, O: AsRef<Path>>(
        &self,
        input_path: P,
        output_dir: O,
    ) -> Result<()> {
        let stems = self.separate_file(input_path)?;
        stems.save_all(output_dir)
    }

    /// Batch separate multiple files
    pub fn separate_batch<P: AsRef<Path>, O: AsRef<Path>>(
        &self,
        input_paths: &[P],
        output_dir: O,
    ) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        for (idx, input_path) in input_paths.iter().enumerate() {
            let input_path = input_path.as_ref();
            let file_stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");

            let file_output = output_dir.join(file_stem);

            if self.config.show_progress {
                log::info!(
                    "Processing file {} of {}: {:?}",
                    idx + 1,
                    input_paths.len(),
                    input_path
                );
            }

            self.separate_and_save(input_path, &file_output)?;
        }

        Ok(())
    }

    /// Get model configuration
    pub fn model_config(&self) -> &ModelConfig {
        &self.config.model
    }

    /// Get processing configuration
    pub fn process_config(&self) -> &ProcessConfig {
        &self.config.process
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separator_config_default() {
        let config = SeparatorConfig::default();
        assert!(config.show_progress);
        assert_eq!(config.model.sample_rate, 44100);
    }

    #[test]
    fn test_stems_creation() {
        let mut sources = HashMap::new();
        let data = ndarray::Array2::zeros((2, 1000));
        sources.insert("vocals".to_string(), AudioBuffer::new(data, 44100));

        let stems = Stems::new(sources);
        assert!(stems.get("vocals").is_some());
        assert!(stems.get("drums").is_none());
    }

    #[test]
    fn test_config_builders() {
        let config = SeparatorConfig::onnx("model.onnx")
            .with_shifts(2)
            .with_segment_length(5.0)
            .with_progress(false);

        assert_eq!(config.process.shifts, 2);
        assert_eq!(config.process.segment_length, Some(5.0));
        assert!(!config.show_progress);
    }
}

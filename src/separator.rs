//! Main separator API

use crate::audio::{AudioBuffer, AudioFile, BitDepth};
use crate::error::{CharonError, Result};
#[cfg(feature = "ort-backend")]
use crate::models::ModelBackend;
use crate::models::{Model, ModelConfig};
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

    /// Create configuration for the 4-stem HTDemucs ONNX export
    /// (see [`ModelConfig::htdemucs`])
    #[cfg(feature = "ort-backend")]
    pub fn htdemucs<P: AsRef<Path>>(model_path: P) -> Self {
        let mut model = ModelConfig::htdemucs(model_path);
        model.backend = Some(ModelBackend::OnnxRuntime);
        Self {
            model,
            ..Self::default()
        }
    }

    /// Configuration for the split-transform HTDemucs export
    /// (see [`ModelConfig::htdemucs_split`])
    #[cfg(feature = "ort-backend")]
    pub fn htdemucs_split<P: AsRef<Path>>(model_path: P) -> Self {
        let mut model = ModelConfig::htdemucs_split(model_path);
        model.backend = Some(ModelBackend::OnnxRuntime);
        Self {
            model,
            ..Self::default()
        }
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

/// Output encoding for saved stems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StemFormat {
    Wav(BitDepth),
    /// FLAC, 16 or 24 bit
    Flac(BitDepth),
}

impl Default for StemFormat {
    fn default() -> Self {
        StemFormat::Wav(BitDepth::Float32)
    }
}

impl StemFormat {
    fn extension(self) -> &'static str {
        match self {
            StemFormat::Wav(_) => "wav",
            StemFormat::Flac(_) => "flac",
        }
    }

    fn write(self, path: &Path, buffer: &AudioBuffer) -> Result<()> {
        match self {
            StemFormat::Wav(depth) => AudioFile::write_wav_with_depth(path, buffer, depth),
            StemFormat::Flac(depth) => AudioFile::write_flac(path, buffer, depth),
        }
    }
}

/// Separated audio stems
pub struct Stems {
    /// Map of source name to audio buffer
    pub sources: HashMap<String, AudioBuffer>,
    /// Source names in model output order
    order: Vec<String>,
}

impl Stems {
    /// Create new stems collection. Names are ordered alphabetically; use
    /// [`Stems::from_ordered`] to keep a specific order.
    pub fn new(sources: HashMap<String, AudioBuffer>) -> Self {
        let mut order: Vec<String> = sources.keys().cloned().collect();
        order.sort();
        Self { sources, order }
    }

    /// Create stems from `(name, buffer)` pairs, keeping their order
    pub fn from_ordered(stems: Vec<(String, AudioBuffer)>) -> Self {
        let order = stems.iter().map(|(name, _)| name.clone()).collect();
        Self {
            sources: stems.into_iter().collect(),
            order,
        }
    }

    /// Get stem by name
    pub fn get(&self, name: &str) -> Option<&AudioBuffer> {
        self.sources.get(name)
    }

    /// Save all stems to directory as 32-bit float WAV
    pub fn save_all<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        self.save_all_as(output_dir, StemFormat::default())
    }

    /// Save all stems to directory in the given format
    pub fn save_all_as<P: AsRef<Path>>(&self, output_dir: P, format: StemFormat) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        for name in &self.order {
            let output_path = output_dir.join(format!("{name}.{}", format.extension()));
            format.write(&output_path, &self.sources[name])?;
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

    /// List stem names in model output order
    pub fn list(&self) -> Vec<String> {
        self.order.clone()
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

        if separated.len() != self.config.model.sources.len() {
            return Err(CharonError::Model(format!(
                "model produced {} sources, config names {}",
                separated.len(),
                self.config.model.sources.len()
            )));
        }
        let stems = self
            .config
            .model
            .sources
            .iter()
            .cloned()
            .zip(separated)
            .collect();

        Ok(Stems::from_ordered(stems))
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

    /// Execution provider the model runs on ("CPU" or "CoreML")
    pub fn provider(&self) -> &'static str {
        match self.model {
            #[cfg(feature = "ort-backend")]
            Model::Onnx(ref m) => m.provider(),
        }
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
    #[cfg(feature = "ort-backend")]
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

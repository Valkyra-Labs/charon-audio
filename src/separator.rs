//! Main separator API

use crate::audio::{AudioBuffer, AudioFile, BitDepth};
use crate::control::Control;
use crate::error::{CharonError, Result};
#[cfg(feature = "ort-backend")]
use crate::models::ModelBackend;
use crate::models::{Model, ModelConfig};
use crate::processor::{ProcessConfig, Processor};
use crate::regions::{self, RegionPlan, RegionWriter, SubSource, REGION_OUTPUTS};
use crate::stream::{AudioSource, StemSink};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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

    /// Configuration for the music branch of TIGER-DnR
    /// (see [`ModelConfig::tiger_music`]): no input normalization (the
    /// network normalizes internally), 12 s windows with half overlap.
    #[cfg(feature = "ort-backend")]
    pub fn tiger_music<P: AsRef<Path>>(model_path: P) -> Self {
        let mut model = ModelConfig::tiger_music(model_path);
        model.backend = Some(ModelBackend::OnnxRuntime);
        Self {
            model,
            process: ProcessConfig {
                segment_length: None,
                overlap: 0.5,
                shifts: 1,
                normalize: false,
                blend: crate::processor::Blend::Triangle,
            },
            ..Self::default()
        }
    }

    /// Set the overlap between model windows, in `[0, 1)`
    pub fn with_overlap(mut self, overlap: f32) -> Self {
        self.process.overlap = overlap;
        self
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
    model: Arc<Model>,
    processor: Processor,
    config: SeparatorConfig,
}

impl Separator {
    /// Create new separator from configuration
    pub fn new(config: SeparatorConfig) -> Result<Self> {
        let model = Arc::new(Model::from_config(config.model.clone())?);
        let processor = Processor::new(config.process.clone());

        Ok(Self {
            model,
            processor,
            config,
        })
    }

    /// A separator with other processing settings (overlap, blend, ...)
    /// on the same loaded model, without loading it again. The two share
    /// one ONNX Runtime session, so their runs take turns; for runs at
    /// the same time, create another separator with [`Separator::new`].
    pub fn with_process_config(&self, process: ProcessConfig) -> Self {
        let mut config = self.config.clone();
        config.process = process.clone();
        Self {
            model: Arc::clone(&self.model),
            processor: Processor::new(process),
            config,
        }
    }

    /// Create separator with default configuration
    pub fn with_default_model() -> Result<Self> {
        Self::new(SeparatorConfig::default())
    }

    /// Separate audio buffer into stems
    pub fn separate(&self, audio: &AudioBuffer) -> Result<Stems> {
        if !self.config.show_progress {
            return self.separate_with(audio, &Control::default());
        }
        let pb = ProgressBar::new(0);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>5}/{len:5} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_message("Separating audio...");
        let bar = pb.clone();
        let control = Control::new().with_progress(move |p| {
            bar.set_length(p.total as u64);
            bar.set_position(p.done as u64);
        });
        let stems = self.separate_with(audio, &control);
        match &stems {
            Ok(_) => pb.finish_with_message("Separation complete!"),
            Err(_) => pb.abandon(),
        }
        stems
    }

    /// Separate audio buffer into stems, reporting progress through
    /// `control` and stopping with [`CharonError::Cancelled`] when it is
    /// cancelled.
    pub fn separate_with(&self, audio: &AudioBuffer, control: &Control) -> Result<Stems> {
        // Resample if needed
        let audio = if audio.sample_rate != self.config.model.sample_rate {
            audio.resample(self.config.model.sample_rate)?
        } else {
            audio.clone()
        };

        // Convert channels if needed (per-channel models take any layout)
        let audio =
            if !self.config.model.per_channel && audio.channels() != self.config.model.channels {
                audio.convert_channels(self.config.model.channels)?
            } else {
                audio
            };

        let separated = self.processor.process_with(&self.model, &audio, control)?;

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

    /// Separate a source of any length into `sink` with bounded memory.
    ///
    /// The source must have the model's sample rate and channel count
    /// (resample and remix before, for example while decoding); time shifts
    /// are not supported. Output is identical to [`Separator::separate_with`]
    /// on the same audio.
    pub fn separate_stream(
        &self,
        source: &mut dyn AudioSource,
        sink: &mut dyn StemSink,
        control: &Control,
    ) -> Result<()> {
        let model = &self.config.model;
        if source.sample_rate() != model.sample_rate
            || (!model.per_channel && source.channels() != model.channels)
        {
            return Err(CharonError::InvalidConfig(format!(
                "stream is {} Hz, {} channels; the model needs {} Hz, {} channels",
                source.sample_rate(),
                source.channels(),
                model.sample_rate,
                model.channels
            )));
        }
        self.processor
            .process_stream(&self.model, source, sink, &model.sources, control)
    }

    /// Reduce one stem inside regions and leave the rest of the input
    /// untouched.
    ///
    /// Writes two stems to `sink`, named by [`REGION_OUTPUTS`]: the
    /// processed mix (equal to the input bit for bit outside the regions)
    /// and the part that was subtracted (silent outside the regions). Each
    /// region is separated with `plan.context` samples of audio around it;
    /// the subtraction fades over `plan.crossfade` samples at the edges.
    /// The source must have the model's sample rate and channel count.
    /// Progress counts the model windows of all regions together.
    pub fn remove_in_regions(
        &self,
        source: &mut dyn AudioSource,
        plan: &RegionPlan,
        sink: &mut dyn StemSink,
        control: &Control,
    ) -> Result<()> {
        let model = &self.config.model;
        if source.sample_rate() != model.sample_rate
            || (!model.per_channel && source.channels() != model.channels)
        {
            return Err(CharonError::InvalidConfig(format!(
                "stream is {} Hz, {} channels; the model needs {} Hz, {} channels",
                source.sample_rate(),
                source.channels(),
                model.sample_rate,
                model.channels
            )));
        }
        let target = model
            .sources
            .iter()
            .position(|name| *name == plan.target)
            .ok_or_else(|| {
                CharonError::InvalidConfig(format!(
                    "the model has no stem named {:?} (it has {:?})",
                    plan.target, model.sources
                ))
            })?;
        let (len, channels, rate) = (source.len(), source.channels(), source.sample_rate());
        let ordered = regions::sorted_regions(plan, len)?;
        let outputs: Vec<String> = REGION_OUTPUTS.iter().map(|s| s.to_string()).collect();
        sink.begin(&outputs, channels, rate)?;

        let spans: Vec<(usize, usize)> = ordered
            .iter()
            .map(|r| regions::span(r, plan.context, len))
            .collect();
        let mut windows = Vec::with_capacity(spans.len());
        for &(a, b) in &spans {
            windows.push(
                self.processor
                    .stream_window_count(&self.model, b - a, rate)?,
            );
        }
        let total: usize = windows.iter().sum();

        let shared = std::cell::RefCell::new(source);
        let mut written = 0usize;
        let mut done_before = 0usize;
        for (i, region) in ordered.iter().enumerate() {
            let (span_start, span_end) = spans[i];
            // This region writes up to the next region's start (or the end
            // of its own span), so neighbours never overwrite each other.
            let write_to = match ordered.get(i + 1) {
                Some(next) => span_end.min(next.start).max(region.end),
                None => span_end,
            };
            if span_start > written {
                regions::copy_through(&shared, sink, channels, written, span_start, control)?;
                written = span_start;
            }
            let mut sub = SubSource {
                inner: &shared,
                offset: span_start,
                len: span_end - span_start,
                channels,
                rate,
            };
            let mut writer = RegionWriter {
                source: &shared,
                sink: &mut *sink,
                region: *region,
                crossfade: plan.crossfade,
                target,
                pos: span_start,
                write_from: written,
                write_to,
            };
            let part = control.part(done_before, total);
            self.processor.process_stream(
                &self.model,
                &mut sub,
                &mut writer,
                &model.sources,
                &part,
            )?;
            done_before += windows[i];
            written = write_to;
        }
        if written < len {
            regions::copy_through(&shared, sink, channels, written, len, control)?;
        }
        sink.finish()
    }

    /// Separate audio from file
    #[cfg(feature = "decode")]
    pub fn separate_file<P: AsRef<Path>>(&self, path: P) -> Result<Stems> {
        let audio = AudioFile::read(path)?;
        self.separate(&audio)
    }

    /// Separate audio and save stems
    #[cfg(feature = "decode")]
    pub fn separate_and_save<P: AsRef<Path>, O: AsRef<Path>>(
        &self,
        input_path: P,
        output_dir: O,
    ) -> Result<()> {
        let stems = self.separate_file(input_path)?;
        stems.save_all(output_dir)
    }

    /// Batch separate multiple files
    #[cfg(feature = "decode")]
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
        match *self.model {
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

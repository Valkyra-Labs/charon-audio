//! Audio processing pipeline

use crate::audio::AudioBuffer;
use crate::error::Result;
use crate::models::Model;
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessConfig {
    /// Segment length in seconds (for splitting long audio)
    pub segment_length: Option<f64>,
    /// Overlap between segments (0.0 to 1.0)
    pub overlap: f32,
    /// Number of shifts for ensemble prediction
    pub shifts: usize,
    /// Normalize input audio
    pub normalize: bool,
    /// Number of parallel jobs (0 = auto)
    pub num_jobs: usize,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            segment_length: Some(10.0),
            overlap: 0.25,
            shifts: 1,
            normalize: true,
            num_jobs: 0,
        }
    }
}

/// Audio processor for source separation
pub struct Processor {
    config: ProcessConfig,
}

impl Processor {
    /// Create new processor
    pub fn new(config: ProcessConfig) -> Self {
        Self { config }
    }

    /// Process audio buffer with model
    pub fn process(&self, model: &Model, audio: &AudioBuffer) -> Result<Vec<AudioBuffer>> {
        let mut processed_audio = audio.clone();

        // Normalize if requested
        if self.config.normalize {
            let mean = processed_audio.data.mean().unwrap_or(0.0);
            let std = processed_audio.data.std(0.0);
            processed_audio
                .data
                .mapv_inplace(|x| (x - mean) / (std + 1e-8));
        }

        // Check if we need to segment
        let segment_samples = self
            .config
            .segment_length
            .map(|len| (len * processed_audio.sample_rate as f64) as usize);

        let separated = if let Some(seg_len) = segment_samples {
            if processed_audio.samples() > seg_len {
                self.process_segmented(model, &processed_audio, seg_len)?
            } else {
                self.process_single(model, &processed_audio)?
            }
        } else {
            self.process_single(model, &processed_audio)?
        };

        // Denormalize and create audio buffers
        let mut output_buffers = Vec::new();
        for separated_source in separated {
            let mut buffer = AudioBuffer::new(separated_source, audio.sample_rate);

            // Apply inverse normalization if needed
            if self.config.normalize {
                let mean = audio.data.mean().unwrap_or(0.0);
                let std = audio.data.std(0.0);
                buffer.data.mapv_inplace(|x| x * (std + 1e-8) + mean);
            }

            output_buffers.push(buffer);
        }

        Ok(output_buffers)
    }

    /// Process single segment
    fn process_single(&self, model: &Model, audio: &AudioBuffer) -> Result<Vec<Array2<f32>>> {
        if self.config.shifts <= 1 {
            model.infer(&audio.data)
        } else {
            self.process_with_shifts(model, audio)
        }
    }

    /// Process with multiple shifts (ensemble)
    fn process_with_shifts(&self, model: &Model, audio: &AudioBuffer) -> Result<Vec<Array2<f32>>> {
        let shift_amount = audio.sample_rate as usize / 2; // 0.5 second shift
        let num_sources = model.config().sources.len();

        let mut accumulated: Vec<Array2<f32>> =
            vec![Array2::zeros((audio.channels(), audio.samples())); num_sources];

        for shift_idx in 0..self.config.shifts {
            let shift = (shift_idx * shift_amount) % audio.samples();

            // Shift input
            let mut shifted_data = audio.data.clone();
            if shift > 0 {
                let (left, right) = shifted_data.view().split_at(ndarray::Axis(1), shift);
                shifted_data = ndarray::concatenate![ndarray::Axis(1), right, left];
            }

            // Run inference
            let separated = model.infer(&shifted_data)?;

            // Shift back and accumulate
            for (src_idx, mut source) in separated.into_iter().enumerate() {
                if shift > 0 {
                    let samples = source.ncols();
                    let unshift = samples - shift;
                    let (left, right) = source.view().split_at(ndarray::Axis(1), unshift);
                    source = ndarray::concatenate![ndarray::Axis(1), right, left];
                }
                accumulated[src_idx] = &accumulated[src_idx] + &source;
            }
        }

        // Average
        for source in &mut accumulated {
            *source /= self.config.shifts as f32;
        }

        Ok(accumulated)
    }

    /// Process audio in segments with overlap
    fn process_segmented(
        &self,
        model: &Model,
        audio: &AudioBuffer,
        segment_length: usize,
    ) -> Result<Vec<Array2<f32>>> {
        let total_samples = audio.samples();
        let overlap_samples = (segment_length as f32 * self.config.overlap) as usize;
        let step = segment_length - overlap_samples;

        // Calculate segments
        let mut segments = Vec::new();
        let mut pos = 0;
        while pos < total_samples {
            let end = (pos + segment_length).min(total_samples);
            segments.push((pos, end));
            pos += step;
            if end >= total_samples {
                break;
            }
        }

        let num_sources = model.config().sources.len();
        let channels = audio.channels();

        // Process segments (can be parallelized)
        let segment_results: Vec<Result<Vec<Array2<f32>>>> = if self.config.num_jobs != 1 {
            segments
                .par_iter()
                .map(|&(start, end)| {
                    let segment = audio.data.slice(ndarray::s![.., start..end]).to_owned();
                    model.infer(&segment)
                })
                .collect()
        } else {
            segments
                .iter()
                .map(|&(start, end)| {
                    let segment = audio.data.slice(ndarray::s![.., start..end]).to_owned();
                    model.infer(&segment)
                })
                .collect()
        };

        // Initialize output arrays
        let mut outputs: Vec<Array2<f32>> =
            vec![Array2::zeros((channels, total_samples)); num_sources];
        let mut weight = Array2::zeros((1, total_samples));

        // Combine segments with overlap
        for (segment_idx, result) in segment_results.into_iter().enumerate() {
            let separated = result?;
            let (start, end) = segments[segment_idx];
            let seg_len = end - start;

            // Create fade in/out for overlap
            let fade = self.create_fade_window(seg_len, overlap_samples);

            for (src_idx, source) in separated.into_iter().enumerate() {
                for ch in 0..channels {
                    for i in 0..seg_len {
                        outputs[src_idx][[ch, start + i]] += source[[ch, i]] * fade[i];
                    }
                }
            }

            // Track weights for normalization
            for i in 0..seg_len {
                weight[[0, start + i]] += fade[i];
            }
        }

        // Normalize by overlap weight
        for output in &mut outputs {
            *output /= &weight;
        }

        Ok(outputs)
    }

    /// Create fade window for overlap-add
    fn create_fade_window(&self, length: usize, overlap: usize) -> Vec<f32> {
        let mut window = vec![1.0; length];

        if overlap > 0 {
            // Fade in
            for (i, win) in window.iter_mut().enumerate().take(overlap.min(length)) {
                let t = i as f32 / overlap as f32;
                *win = t;
            }

            // Fade out
            for i in 0..overlap.min(length) {
                let idx = length - overlap + i;
                if idx < length {
                    let t = i as f32 / overlap as f32;
                    window[idx] = 1.0 - t;
                }
            }
        }

        window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_config_default() {
        let config = ProcessConfig::default();
        assert_eq!(config.overlap, 0.25);
        assert_eq!(config.shifts, 1);
        assert!(config.normalize);
    }

    #[test]
    fn test_fade_window() {
        use approx::assert_abs_diff_eq;

        let processor = Processor::new(ProcessConfig::default());
        let window = processor.create_fade_window(100, 20);

        assert_eq!(window.len(), 100);
        // First sample starts fading in from 0
        assert_abs_diff_eq!(window[0], 0.0, epsilon = 0.01);
        // Last sample is fading out (near 0 but not exactly 0)
        assert!(window[99] < 0.1);
        // Middle should be at full volume
        assert_abs_diff_eq!(window[50], 1.0, epsilon = 0.01);
    }
}

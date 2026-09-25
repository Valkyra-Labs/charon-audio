//! Audio processing pipeline
//!
//! Segmentation, overlap-add, time-shift ensembling and input normalization
//! follow Demucs `apply_model` / `Separator.separate_tensor` (demucs 4.1.0),
//! so that a model exported from Demucs produces the same output here as in
//! the reference implementation.

use crate::audio::AudioBuffer;
use crate::error::{CharonError, Result};
use crate::models::Model;
use ndarray::{s, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessConfig {
    /// Segment length in seconds. Ignored when the model has a fixed input
    /// length (`ModelConfig::segment_samples`), which then takes precedence.
    pub segment_length: Option<f64>,
    /// Overlap between segments (0.0 to 1.0)
    pub overlap: f32,
    /// Number of time shifts to average. `0` or `1` disables shifting;
    /// this matches Demucs `shifts=0`, not Demucs `shifts=1` (one random shift).
    pub shifts: usize,
    /// Normalize input by the mono reference mean/std (Demucs convention)
    pub normalize: bool,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            segment_length: Some(10.0),
            overlap: 0.25,
            shifts: 1,
            normalize: true,
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
        if audio.samples() == 0 {
            return Err(CharonError::Audio("Input audio is empty".to_string()));
        }
        if !(0.0..1.0).contains(&self.config.overlap) {
            return Err(CharonError::InvalidConfig(format!(
                "overlap must be in [0, 1), got {}",
                self.config.overlap
            )));
        }

        let (mean, std) = if self.config.normalize {
            reference_stats(audio.data.view())
        } else {
            (0.0, 1.0)
        };
        let input = audio.data.mapv(|x| (x - mean) / std);

        let segment = self.segment_samples(model, audio.sample_rate)?;
        let separated = if self.config.shifts > 1 {
            self.process_shifted(model, input.view(), segment, audio.sample_rate)?
        } else {
            self.process_split(model, input.view(), segment)?
        };

        Ok(separated
            .into_iter()
            .map(|mut source| {
                source.mapv_inplace(|x| x * std + mean);
                AudioBuffer::new(source, audio.sample_rate)
            })
            .collect())
    }

    fn segment_samples(&self, model: &Model, sample_rate: u32) -> Result<Option<usize>> {
        if let Some(fixed) = model.config().segment_samples {
            return Ok(Some(fixed));
        }
        match self.config.segment_length {
            Some(seconds) if seconds > 0.0 => Ok(Some((seconds * sample_rate as f64) as usize)),
            Some(seconds) => Err(CharonError::InvalidConfig(format!(
                "segment_length must be positive, got {seconds}"
            ))),
            None => Ok(None),
        }
    }

    /// Average predictions over time-shifted copies of the input.
    ///
    /// The input is zero-padded by half a second on both sides, so the shift
    /// never wraps audio around. Offsets are deterministic and evenly spaced in
    /// `[0, max_shift)` (Demucs draws them at random).
    fn process_shifted(
        &self,
        model: &Model,
        input: ArrayView2<f32>,
        segment: Option<usize>,
        sample_rate: u32,
    ) -> Result<Vec<Array2<f32>>> {
        let (channels, length) = input.dim();
        let shifts = self.config.shifts;
        let max_shift = sample_rate as usize / 2;

        let mut padded = Array2::zeros((channels, length + 2 * max_shift));
        padded
            .slice_mut(s![.., max_shift..max_shift + length])
            .assign(&input);

        let mut accumulated: Option<Vec<Array2<f32>>> = None;
        for shift_idx in 0..shifts {
            let offset = shift_idx * max_shift / shifts;
            let shifted = padded.slice(s![.., offset..offset + length + max_shift - offset]);
            let separated = self.process_split(model, shifted, segment)?;

            let acc = accumulated
                .get_or_insert_with(|| vec![Array2::zeros((channels, length)); separated.len()]);
            for (acc_source, source) in acc.iter_mut().zip(&separated) {
                *acc_source += &source.slice(s![.., max_shift - offset..]);
            }
        }

        let mut out = accumulated.unwrap_or_default();
        for source in &mut out {
            *source /= shifts as f32;
        }
        Ok(out)
    }

    /// Split into overlapping segments, run the model on each and blend them
    /// with triangular weights (Demucs `apply_model` with `split=True`).
    fn process_split(
        &self,
        model: &Model,
        input: ArrayView2<f32>,
        segment: Option<usize>,
    ) -> Result<Vec<Array2<f32>>> {
        let (channels, length) = input.dim();
        let Some(segment) = segment else {
            return self.run_window(model, input, length, 0, length);
        };

        let stride = (((1.0 - self.config.overlap) * segment as f32) as usize).max(1);
        let weight = triangle_weight(segment);

        let mut out: Option<Vec<Array2<f32>>> = None;
        let mut sum_weight = vec![0.0f32; length];

        for offset in (0..length).step_by(stride) {
            let chunk_len = segment.min(length - offset);
            let chunk_out = self.run_window(model, input, segment, offset, chunk_len)?;

            let out =
                out.get_or_insert_with(|| vec![Array2::zeros((channels, length)); chunk_out.len()]);
            let w = &weight[..chunk_len];
            for (dst, src) in out.iter_mut().zip(&chunk_out) {
                for ch in 0..channels {
                    let mut dst_row = dst.slice_mut(s![ch, offset..offset + chunk_len]);
                    let src_row = src.row(ch);
                    for ((d, &x), &wi) in dst_row.iter_mut().zip(src_row.iter()).zip(w) {
                        *d += x * wi;
                    }
                }
            }
            for (acc, &wi) in sum_weight[offset..offset + chunk_len].iter_mut().zip(w) {
                *acc += wi;
            }
        }

        let mut out = out.unwrap_or_default();
        for source in &mut out {
            for mut row in source.rows_mut() {
                for (x, &wsum) in row.iter_mut().zip(&sum_weight) {
                    *x /= wsum;
                }
            }
        }
        Ok(out)
    }

    /// Run the model on `input[offset..offset + chunk_len]`, centred inside a
    /// window of `window` samples. Samples of `input` around the chunk fill the
    /// window where available and zeros elsewhere; the model output is then
    /// centre-trimmed back to `chunk_len` (Demucs `TensorChunk.padded` +
    /// `center_trim`).
    fn run_window(
        &self,
        model: &Model,
        input: ArrayView2<f32>,
        window: usize,
        offset: usize,
        chunk_len: usize,
    ) -> Result<Vec<Array2<f32>>> {
        let (channels, total) = input.dim();
        let delta = window - chunk_len;
        let start = offset as isize - (delta / 2) as isize;
        let end = start + window as isize;
        let src_start = start.max(0) as usize;
        let src_end = (end.min(total as isize)) as usize;
        let dst_start = (src_start as isize - start) as usize;

        let mut padded = Array2::zeros((channels, window));
        padded
            .slice_mut(s![.., dst_start..dst_start + (src_end - src_start)])
            .assign(&input.slice(s![.., src_start..src_end]));

        let trim = delta / 2;
        model
            .infer(&padded)?
            .into_iter()
            .map(|source| {
                if source.dim() != (channels, window) {
                    return Err(CharonError::Model(format!(
                        "model returned shape {:?}, expected ({channels}, {window})",
                        source.dim()
                    )));
                }
                Ok(source.slice(s![.., trim..trim + chunk_len]).to_owned())
            })
            .collect()
    }
}

/// Mean and standard deviation of the mono mix, as used by Demucs to
/// normalize its input. The deviation uses Bessel's correction (like
/// `torch.std`) plus 1e-8, so silence does not divide by zero.
fn reference_stats(data: ArrayView2<f32>) -> (f32, f32) {
    let mono = data
        .mean_axis(ndarray::Axis(0))
        .expect("audio has at least one channel");
    let n = mono.len();
    let mean = mono.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var = if n > 1 {
        mono.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    (mean as f32, (var.sqrt() + 1e-8) as f32)
}

/// Triangular blending weight with its peak in the middle, normalized to a
/// maximum of 1. Every sample has non-zero weight, so overlap-add never
/// divides by zero.
fn triangle_weight(segment: usize) -> Vec<f32> {
    let half = segment / 2;
    let rising = (1..=half).map(|i| i as f32);
    let falling = (1..=segment - half).rev().map(|i| i as f32);
    let weight: Vec<f32> = rising.chain(falling).collect();
    let max = weight.iter().copied().fold(0.0f32, f32::max);
    weight.into_iter().map(|w| w / max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_process_config_default() {
        let config = ProcessConfig::default();
        assert_eq!(config.overlap, 0.25);
        assert_eq!(config.shifts, 1);
        assert!(config.normalize);
    }

    #[test]
    fn test_triangle_weight_matches_demucs() {
        // demucs: cat(arange(1, seg//2 + 1), arange(seg - seg//2, 0, -1)) / max
        let w = triangle_weight(7);
        let expected = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0].map(|x| x / 4.0);
        assert_eq!(w.len(), 7);
        for (a, b) in w.iter().zip(expected) {
            assert_abs_diff_eq!(*a, b, epsilon = 1e-7);
        }
        assert!(triangle_weight(343_980).iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_reference_stats() {
        // Channels average to [1, 2, 3, 4]: mean 2.5, unbiased std sqrt(5/3).
        let data = Array2::from_shape_vec((2, 4), vec![0., 2., 2., 4., 2., 2., 4., 4.]).unwrap();
        let (mean, std) = reference_stats(data.view());
        assert_abs_diff_eq!(mean, 2.5, epsilon = 1e-6);
        assert_abs_diff_eq!(std, (5.0f32 / 3.0).sqrt(), epsilon = 1e-6);
    }
}

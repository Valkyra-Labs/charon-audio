//! Audio processing pipeline
//!
//! Segmentation, overlap-add, time-shift ensembling and input normalization
//! follow Demucs `apply_model` / `Separator.separate_tensor` (demucs 4.1.0),
//! so that a model exported from Demucs produces the same output here as in
//! the reference implementation.

use crate::audio::AudioBuffer;
use crate::control::{Control, Progress};
use crate::error::{CharonError, Result};
use crate::models::Model;
use crate::stream::{AudioSource, StemSink};
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
    /// How overlapping windows are blended
    #[serde(default)]
    pub blend: Blend,
}

/// Weighting of overlapping model windows in overlap-add.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Blend {
    /// Triangular weights peaking mid-window (Demucs `apply_model`)
    #[default]
    Triangle,
    /// Equal weights: a plain average of the windows covering a sample
    /// (TIGER's `wav_chunk_inference`)
    Uniform,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            segment_length: Some(10.0),
            overlap: 0.25,
            shifts: 1,
            normalize: true,
            blend: Blend::Triangle,
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
        self.process_with(model, audio, &Control::default())
    }

    /// Process audio buffer with model, reporting progress per model window
    /// and stopping with [`CharonError::Cancelled`] when `control` is
    /// cancelled.
    pub fn process_with(
        &self,
        model: &Model,
        audio: &AudioBuffer,
        control: &Control,
    ) -> Result<Vec<AudioBuffer>> {
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
        let mut tracker = Tracker {
            control,
            done: 0,
            total: self.window_count(audio.samples(), segment, audio.sample_rate),
        };
        tracker.check()?;
        let separated = if self.config.shifts > 1 {
            self.process_shifted(
                model,
                input.view(),
                segment,
                audio.sample_rate,
                &mut tracker,
            )?
        } else {
            self.process_split(model, input.view(), segment, &mut tracker)?
        };

        Ok(separated
            .into_iter()
            .map(|mut source| {
                source.mapv_inplace(|x| x * std + mean);
                AudioBuffer::new(source, audio.sample_rate)
            })
            .collect())
    }

    /// Number of model windows a streaming job over `length` samples runs.
    pub(crate) fn stream_window_count(
        &self,
        model: &Model,
        length: usize,
        sample_rate: u32,
    ) -> Result<usize> {
        let segment = self.segment_samples(model, sample_rate)?;
        Ok(self.window_count(length, segment, sample_rate))
    }

    /// Number of model windows a job over `length` samples runs.
    fn window_count(&self, length: usize, segment: Option<usize>, sample_rate: u32) -> usize {
        let per_pass = |len: usize| match segment {
            None => 1,
            Some(segment) => {
                let stride = self.stride(segment);
                len.div_ceil(stride)
            }
        };
        if self.config.shifts > 1 {
            let max_shift = sample_rate as usize / 2;
            (0..self.config.shifts)
                .map(|i| {
                    let offset = i * max_shift / self.config.shifts;
                    per_pass(length + max_shift - offset)
                })
                .sum()
        } else {
            per_pass(length)
        }
    }

    fn window_weight(&self, segment: usize) -> Vec<f32> {
        match self.config.blend {
            Blend::Triangle => triangle_weight(segment),
            Blend::Uniform => vec![1.0; segment],
        }
    }

    /// Window stride in samples: `(1 - overlap) * segment`, computed in
    /// f64 and rounded, so that an overlap of 2/3 on 529200 samples gives
    /// exactly 176400 (f32 and truncation gave 176399, which shifts every
    /// window by one more sample than the previous one).
    fn stride(&self, segment: usize) -> usize {
        (((1.0 - self.config.overlap as f64) * segment as f64).round() as usize).max(1)
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
        tracker: &mut Tracker,
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
            let separated = self.process_split(model, shifted, segment, tracker)?;

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
        tracker: &mut Tracker,
    ) -> Result<Vec<Array2<f32>>> {
        let (channels, length) = input.dim();
        let Some(segment) = segment else {
            let out = self.run_window(model, input, length, 0, length)?;
            tracker.step()?;
            return Ok(out);
        };

        let stride = self.stride(segment);
        let weight = self.window_weight(segment);

        let mut out: Option<Vec<Array2<f32>>> = None;
        let mut sum_weight = vec![0.0f32; length];

        for offset in (0..length).step_by(stride) {
            let chunk_len = segment.min(length - offset);
            let chunk_out = self.run_window(model, input, segment, offset, chunk_len)?;
            tracker.step()?;

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

        self.infer_trimmed(model, &padded, window, chunk_len)
    }

    /// Run the model on a full window and centre-trim every source back to
    /// `chunk_len` samples.
    fn infer_trimmed(
        &self,
        model: &Model,
        padded: &Array2<f32>,
        window: usize,
        chunk_len: usize,
    ) -> Result<Vec<Array2<f32>>> {
        let channels = padded.nrows();
        let trim = (window - chunk_len) / 2;
        model
            .infer(padded)?
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

    /// Separate a source of any length into `sink` with bounded memory.
    ///
    /// Output is identical to [`Processor::process_with`] on the same input.
    /// The source must already have the model's sample rate and channel
    /// count. Time shifts (`shifts > 1`) are not supported here.
    pub fn process_stream(
        &self,
        model: &Model,
        source: &mut dyn AudioSource,
        sink: &mut dyn StemSink,
        stems: &[String],
        control: &Control,
    ) -> Result<()> {
        let length = source.len();
        let channels = source.channels();
        let rate = source.sample_rate();
        if length == 0 {
            return Err(CharonError::Audio("Input audio is empty".to_string()));
        }
        if !(0.0..1.0).contains(&self.config.overlap) {
            return Err(CharonError::InvalidConfig(format!(
                "overlap must be in [0, 1), got {}",
                self.config.overlap
            )));
        }
        if self.config.shifts > 1 {
            return Err(CharonError::NotSupported(
                "time shifts in streaming separation".to_string(),
            ));
        }

        let segment = self.segment_samples(model, rate)?;
        let Some(segment) = segment else {
            // One window over the whole input: nothing to stream.
            let mut data = Array2::zeros((channels, length));
            source.read(0, data.view_mut())?;
            let separated = self.process_with(model, &AudioBuffer::new(data, rate), control)?;
            check_stem_count(stems, separated.len())?;
            sink.begin(stems, channels, rate)?;
            for (i, stem) in separated.iter().enumerate() {
                sink.write(i, stem.data.view())?;
            }
            return sink.finish();
        };

        let (mean, std) = if self.config.normalize {
            stream_stats(source)?
        } else {
            (0.0, 1.0)
        };
        let mut tracker = Tracker {
            control,
            done: 0,
            total: self.window_count(length, Some(segment), rate),
        };
        tracker.check()?;
        sink.begin(stems, channels, rate)?;

        let stride = self.stride(segment);
        let weight = self.window_weight(segment);
        // Pending (not yet final) output, starting at sample `emitted`.
        let mut acc: Vec<Array2<f32>> = vec![Array2::zeros((channels, 0)); stems.len()];
        let mut sum_weight: Vec<f32> = Vec::new();
        let mut emitted = 0usize;

        for offset in (0..length).step_by(stride) {
            let chunk_len = segment.min(length - offset);
            let padded = read_window(source, mean, std, segment, offset, chunk_len)?;
            let chunk_out = self.infer_trimmed(model, &padded, segment, chunk_len)?;
            tracker.step()?;
            check_stem_count(stems, chunk_out.len())?;

            let rel = offset - emitted;
            let need = rel + chunk_len;
            if sum_weight.len() < need {
                sum_weight.resize(need, 0.0);
                for a in &mut acc {
                    let mut grown = Array2::zeros((channels, need));
                    grown.slice_mut(s![.., ..a.ncols()]).assign(a);
                    *a = grown;
                }
            }
            let w = &weight[..chunk_len];
            for (dst, src) in acc.iter_mut().zip(&chunk_out) {
                for ch in 0..channels {
                    let mut dst_row = dst.slice_mut(s![ch, rel..rel + chunk_len]);
                    let src_row = src.row(ch);
                    for ((d, &x), &wi) in dst_row.iter_mut().zip(src_row.iter()).zip(w) {
                        *d += x * wi;
                    }
                }
            }
            for (a, &wi) in sum_weight[rel..rel + chunk_len].iter_mut().zip(w) {
                *a += wi;
            }

            // Windows start at multiples of `stride`, so nothing after this
            // one touches samples before `offset + stride`.
            let final_upto = (offset + stride).min(length);
            let k = final_upto - emitted;
            for (i, a) in acc.iter_mut().enumerate() {
                let mut block = a.slice(s![.., ..k]).to_owned();
                for mut row in block.rows_mut() {
                    for (x, &wsum) in row.iter_mut().zip(&sum_weight[..k]) {
                        *x /= wsum;
                    }
                }
                block.mapv_inplace(|x| x * std + mean);
                sink.write(i, block.view())?;
                *a = a.slice(s![.., k..]).to_owned();
            }
            sum_weight.drain(..k);
            emitted = final_upto;
        }
        sink.finish()
    }
}

fn check_stem_count(stems: &[String], produced: usize) -> Result<()> {
    if produced != stems.len() {
        return Err(CharonError::Model(format!(
            "model produced {produced} sources, config names {}",
            stems.len()
        )));
    }
    Ok(())
}

/// Read `source[offset..offset + chunk_len]` normalized and centred in a
/// window of `window` samples, real audio around the chunk where the
/// source has it and zeros elsewhere (the streaming twin of
/// `Processor::run_window`).
fn read_window(
    source: &mut dyn AudioSource,
    mean: f32,
    std: f32,
    window: usize,
    offset: usize,
    chunk_len: usize,
) -> Result<Array2<f32>> {
    let channels = source.channels();
    let total = source.len();
    let delta = window - chunk_len;
    let start = offset as isize - (delta / 2) as isize;
    let end = start + window as isize;
    let src_start = start.max(0) as usize;
    let src_end = (end.min(total as isize)) as usize;
    let dst_start = (src_start as isize - start) as usize;

    let mut padded = Array2::zeros((channels, window));
    let mut region = padded.slice_mut(s![.., dst_start..dst_start + (src_end - src_start)]);
    source.read(src_start, region.view_mut())?;
    region.mapv_inplace(|x| (x - mean) / std);
    Ok(padded)
}

/// [`reference_stats`] over a source read in blocks, with the same
/// arithmetic in the same order, so the result is bit-identical.
fn stream_stats(source: &mut dyn AudioSource) -> Result<(f32, f32)> {
    const BLOCK: usize = 1 << 16;
    let (channels, n) = (source.channels(), source.len());
    let mut buf = Array2::zeros((channels, BLOCK.min(n)));
    let mut mono_blocks = |source: &mut dyn AudioSource, f: &mut dyn FnMut(f32)| -> Result<()> {
        let mut start = 0;
        while start < n {
            let len = BLOCK.min(n - start);
            let mut view = buf.slice_mut(s![.., ..len]);
            source.read(start, view.view_mut())?;
            let mono = view
                .mean_axis(ndarray::Axis(0))
                .expect("audio has at least one channel");
            for &x in mono.iter() {
                f(x);
            }
            start += len;
        }
        Ok(())
    };
    let mut sum = 0.0f64;
    mono_blocks(source, &mut |x| sum += x as f64)?;
    let mean = sum / n as f64;
    let mut sq = 0.0f64;
    if n > 1 {
        mono_blocks(source, &mut |x| sq += (x as f64 - mean).powi(2))?;
    }
    let var = if n > 1 { sq / (n - 1) as f64 } else { 0.0 };
    Ok((mean as f32, (var.sqrt() + 1e-8) as f32))
}

/// Counts finished windows, reports them and checks for cancellation.
struct Tracker<'a> {
    control: &'a Control,
    done: usize,
    total: usize,
}

impl Tracker<'_> {
    fn check(&self) -> Result<()> {
        if self.control.is_cancelled() {
            Err(CharonError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// One window finished: report it, then stop if cancelled.
    fn step(&mut self) -> Result<()> {
        self.done += 1;
        self.control.report(Progress {
            done: self.done,
            total: self.total,
        });
        self.check()
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

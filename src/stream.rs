//! Streaming input and output for files longer than memory.
//!
//! A three-hour stereo track is 3.8 GB per stem as `f32`; holding the input
//! and four stems at once does not fit a laptop. [`AudioSource`] gives the
//! processor random access to the input in blocks, and [`StemSink`]
//! receives the finished stems in order, so memory stays bounded by a few
//! model windows regardless of the file length.
//!
//! Streaming output is sample-for-sample identical to [`Processor::process`]
//! on the same input (tested), because it runs the same windows in the same
//! order and blends them with the same arithmetic.
//!
//! [`Processor::process`]: crate::processor::Processor::process

use crate::audio::AudioBuffer;
use crate::error::{CharonError, Result};
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};

/// Random-access audio input.
///
/// The processor reads the input twice: once for the normalization
/// statistics, once for the model windows (with context around each
/// window, so reads overlap).
pub trait AudioSource {
    /// Number of channels.
    fn channels(&self) -> usize;
    /// Sample rate in Hz.
    fn sample_rate(&self) -> u32;
    /// Length in samples per channel.
    fn len(&self) -> usize;
    /// Whether the source has no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Fill `out` (`[channels, n]`) with samples `start..start + n`.
    /// The range is always inside `0..len()`.
    fn read(&mut self, start: usize, out: ArrayViewMut2<f32>) -> Result<()>;
}

/// Sequential stem output.
pub trait StemSink {
    /// Called once before any block, with the stem names in model order,
    /// the channel count and the sample rate.
    fn begin(&mut self, stems: &[String], channels: usize, sample_rate: u32) -> Result<()>;
    /// The next block of one stem (`[channels, n]`). Blocks of every stem
    /// arrive in time order and cover the whole input exactly once.
    fn write(&mut self, stem: usize, block: ArrayView2<f32>) -> Result<()>;
    /// Called once after the last block.
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

/// An in-memory [`AudioSource`] over an [`AudioBuffer`].
pub struct BufferSource<'a> {
    audio: &'a AudioBuffer,
}

impl<'a> BufferSource<'a> {
    /// Read from `audio`.
    pub fn new(audio: &'a AudioBuffer) -> Self {
        Self { audio }
    }
}

impl AudioSource for BufferSource<'_> {
    fn channels(&self) -> usize {
        self.audio.channels()
    }
    fn sample_rate(&self) -> u32 {
        self.audio.sample_rate
    }
    fn len(&self) -> usize {
        self.audio.samples()
    }
    fn read(&mut self, start: usize, mut out: ArrayViewMut2<f32>) -> Result<()> {
        let n = out.ncols();
        if start + n > self.len() || out.nrows() != self.channels() {
            return Err(CharonError::Processing(format!(
                "read {start}..{} of {} samples, {} channels requested of {}",
                start + n,
                self.len(),
                out.nrows(),
                self.channels()
            )));
        }
        out.assign(&self.audio.data.slice(s![.., start..start + n]));
        Ok(())
    }
}

/// An in-memory [`StemSink`] that collects every stem.
#[derive(Debug, Default)]
pub struct BufferSink {
    names: Vec<String>,
    sample_rate: u32,
    stems: Vec<Vec<Array2<f32>>>,
}

impl BufferSink {
    /// An empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// The collected stems as `(name, buffer)` in model order.
    pub fn into_stems(self) -> Vec<(String, AudioBuffer)> {
        let rate = self.sample_rate;
        self.names
            .into_iter()
            .zip(self.stems)
            .map(|(name, blocks)| {
                let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                let data = if views.is_empty() {
                    Array2::zeros((0, 0))
                } else {
                    ndarray::concatenate(ndarray::Axis(1), &views)
                        .expect("blocks of one stem share the channel count")
                };
                (name, AudioBuffer::new(data, rate))
            })
            .collect()
    }
}

impl StemSink for BufferSink {
    fn begin(&mut self, stems: &[String], _channels: usize, sample_rate: u32) -> Result<()> {
        self.names = stems.to_vec();
        self.sample_rate = sample_rate;
        self.stems = vec![Vec::new(); stems.len()];
        Ok(())
    }
    fn write(&mut self, stem: usize, block: ArrayView2<f32>) -> Result<()> {
        self.stems[stem].push(block.to_owned());
        Ok(())
    }
}

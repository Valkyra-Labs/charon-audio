//! Separation limited to regions, with the target stem subtracted.
//!
//! The music-removal job: inside marked regions, subtract (part of) one
//! stem from the mix; outside them, leave the input untouched sample for
//! sample. Subtracting the estimate from the mix, instead of summing the
//! other stems, keeps everything the model did not attribute to the target
//! exactly as it was, including what the other stems would have lost to
//! separation error.
//!
//! Each region is separated with `context` samples of real audio on both
//! sides, and the subtraction fades in and out over `crossfade` samples at
//! the region edges, so there is no step in the output.

use crate::control::Control;
use crate::error::{CharonError, Result};
use crate::stream::{AudioSource, StemSink};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use std::cell::RefCell;

/// A span of the input where the target stem is reduced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    /// First sample of the region.
    pub start: usize,
    /// One past the last sample.
    pub end: usize,
    /// Gain kept on the target stem inside the region: `0.0` removes it,
    /// `1.0` keeps it (the region then changes nothing).
    pub keep: f32,
}

/// What to remove and where.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionPlan {
    /// Name of the stem to reduce, as the model names it (e.g. "music").
    pub target: String,
    /// Regions, in any order; they must not overlap.
    pub regions: Vec<Region>,
    /// Real audio added on both sides of a region before separating it.
    pub context: usize,
    /// Length of the fade in and out of the subtraction at region edges.
    pub crossfade: usize,
}

/// Names of the two output stems of a region job, in order: the processed
/// mix, and what was subtracted from it (silent outside regions).
pub const REGION_OUTPUTS: [&str; 2] = ["result", "removed"];

/// Validate and sort the regions of a plan for an input of `len` samples.
pub(crate) fn sorted_regions(plan: &RegionPlan, len: usize) -> Result<Vec<Region>> {
    let mut regions = plan.regions.clone();
    regions.sort_by_key(|r| r.start);
    for r in &regions {
        if r.start >= r.end || r.end > len {
            return Err(CharonError::InvalidConfig(format!(
                "region {}..{} is empty or outside the input of {len} samples",
                r.start, r.end
            )));
        }
        if !(0.0..=1.0).contains(&r.keep) {
            return Err(CharonError::InvalidConfig(format!(
                "region keep gain must be in [0, 1], got {}",
                r.keep
            )));
        }
    }
    for pair in regions.windows(2) {
        if pair[1].start < pair[0].end {
            return Err(CharonError::InvalidConfig(format!(
                "regions {}..{} and {}..{} overlap",
                pair[0].start, pair[0].end, pair[1].start, pair[1].end
            )));
        }
    }
    Ok(regions)
}

/// Weight of the subtraction at absolute sample `t` of `region`: rises
/// linearly over `crossfade` samples after the start, falls over
/// `crossfade` samples before the end, 1 in between, never 0 inside.
pub(crate) fn fade(region: &Region, crossfade: usize, t: usize) -> f32 {
    let ramp = (crossfade + 1) as f32;
    let rise = (t - region.start + 1) as f32 / ramp;
    let fall = (region.end - t) as f32 / ramp;
    rise.min(fall).min(1.0)
}

/// A window of a shared source, `offset..offset + len`, as its own source.
pub(crate) struct SubSource<'a, 'b> {
    pub(crate) inner: &'a RefCell<&'b mut dyn AudioSource>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) channels: usize,
    pub(crate) rate: u32,
}

impl AudioSource for SubSource<'_, '_> {
    fn channels(&self) -> usize {
        self.channels
    }
    fn sample_rate(&self) -> u32 {
        self.rate
    }
    fn len(&self) -> usize {
        self.len
    }
    fn read(&mut self, start: usize, out: ArrayViewMut2<f32>) -> Result<()> {
        self.inner.borrow_mut().read(self.offset + start, out)
    }
}

/// Receives the separated stems of one region span and writes the region
/// job's outputs (processed mix and removed part) for the absolute range
/// `write_from..write_to` to the outer sink; the rest of the span (context
/// that belongs to a neighbour or was already written) is dropped.
pub(crate) struct RegionWriter<'a, 'b, 'c> {
    pub(crate) source: &'a RefCell<&'b mut dyn AudioSource>,
    pub(crate) sink: &'c mut dyn StemSink,
    pub(crate) region: Region,
    pub(crate) crossfade: usize,
    /// Index of the target among the model stems.
    pub(crate) target: usize,
    /// Absolute position of the next incoming block of the target stem.
    pub(crate) pos: usize,
    pub(crate) write_from: usize,
    pub(crate) write_to: usize,
}

impl StemSink for RegionWriter<'_, '_, '_> {
    fn begin(&mut self, _stems: &[String], _channels: usize, _sample_rate: u32) -> Result<()> {
        Ok(())
    }

    fn write(&mut self, stem: usize, block: ArrayView2<f32>) -> Result<()> {
        if stem != self.target {
            return Ok(());
        }
        let (channels, n) = block.dim();
        let block_start = self.pos;
        self.pos += n;
        let from = block_start.max(self.write_from);
        let to = (block_start + n).min(self.write_to);
        if from >= to {
            return Ok(());
        }
        let len = to - from;
        let mut result = Array2::zeros((channels, len));
        self.source.borrow_mut().read(from, result.view_mut())?;
        let mut removed = Array2::zeros((channels, len));
        let cut = 1.0 - self.region.keep;
        for i in 0..len {
            let t = from + i;
            if t < self.region.start || t >= self.region.end {
                continue;
            }
            let w = fade(&self.region, self.crossfade, t) * cut;
            for ch in 0..channels {
                removed[[ch, i]] = w * block[[ch, t - block_start]];
            }
        }
        // Outside the region `removed` is exactly zero and `x - 0.0` is `x`,
        // so those samples stay the input bit for bit.
        result -= &removed;
        self.sink.write(0, result.view())?;
        self.sink.write(1, removed.view())?;
        Ok(())
    }
}

/// Copy `start..end` of the source to the outputs unchanged (processed mix
/// is the input, removed part is silence), in blocks.
pub(crate) fn copy_through(
    source: &RefCell<&mut dyn AudioSource>,
    sink: &mut dyn StemSink,
    channels: usize,
    start: usize,
    end: usize,
    control: &Control,
) -> Result<()> {
    const BLOCK: usize = 1 << 16;
    let mut pos = start;
    while pos < end {
        if control.cancel_token().is_cancelled() {
            return Err(CharonError::Cancelled);
        }
        let n = BLOCK.min(end - pos);
        let mut block = Array2::zeros((channels, n));
        source.borrow_mut().read(pos, block.view_mut())?;
        sink.write(0, block.view())?;
        sink.write(1, Array2::<f32>::zeros((channels, n)).view())?;
        pos += n;
    }
    Ok(())
}

/// Clamp a region to its span with context inside `0..len`.
pub(crate) fn span(region: &Region, context: usize, len: usize) -> (usize, usize) {
    (
        region.start.saturating_sub(context),
        (region.end + context).min(len),
    )
}

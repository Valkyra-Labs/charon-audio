//! Performance optimization utilities inspired by rust-imbalanced-learn

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Performance hints for audio processing optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceHint {
    /// Prefer cache-friendly access patterns
    CacheFriendly,
    /// Use SIMD when possible
    Vectorize,
    /// Parallelize operations
    Parallel,
    /// Use GPU acceleration if available
    GpuAccelerated,
    /// Optimize for low latency
    LowLatency,
    /// Optimize for throughput
    HighThroughput,
}

/// Collection of performance hints
#[derive(Debug, Clone, Default)]
pub struct PerformanceHints {
    hints: Vec<PerformanceHint>,
}

impl PerformanceHints {
    /// Create new performance hints
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a performance hint
    pub fn with_hint(mut self, hint: PerformanceHint) -> Self {
        self.hints.push(hint);
        self
    }

    /// Check if hint is present
    pub fn has_hint(&self, hint: PerformanceHint) -> bool {
        self.hints.contains(&hint)
    }

    /// Get all hints
    pub fn hints(&self) -> &[PerformanceHint] {
        &self.hints
    }
}

/// KNN-based utilities for audio similarity and feature extraction
pub struct AudioKNN {
    k_neighbors: usize,
}

impl AudioKNN {
    /// Create new AudioKNN with specified number of neighbors
    pub fn new(k_neighbors: usize) -> Self {
        Self { k_neighbors }
    }

    /// Find k nearest neighbors using Euclidean distance
    /// Returns indices and distances of nearest neighbors
    pub fn find_neighbors(
        &self,
        query: ArrayView2<f32>,
        data: ArrayView2<f32>,
    ) -> Vec<Vec<(usize, f32)>> {
        let n_queries = query.nrows();
        let n_data = data.nrows();

        (0..n_queries)
            .into_par_iter()
            .map(|q_idx| {
                let query_point = query.row(q_idx);
                let mut distances: Vec<(usize, f32)> = (0..n_data)
                    .map(|d_idx| {
                        let data_point = data.row(d_idx);
                        let dist = Self::euclidean_distance(query_point, data_point);
                        (d_idx, dist)
                    })
                    .collect();

                // Sort by distance and take k nearest
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                distances.truncate(self.k_neighbors);
                distances
            })
            .collect()
    }

    /// SIMD-optimized Euclidean distance calculation
    #[inline(always)]
    fn euclidean_distance(a: ndarray::ArrayView1<f32>, b: ndarray::ArrayView1<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Find similar audio segments within a larger audio buffer
    pub fn find_similar_segments(
        &self,
        segment: ArrayView2<f32>,
        audio: ArrayView2<f32>,
        hop_size: usize,
    ) -> Vec<(usize, f32)> {
        let segment_len = segment.ncols();
        let audio_len = audio.ncols();

        if segment_len > audio_len {
            return Vec::new();
        }

        let num_windows = (audio_len - segment_len) / hop_size + 1;

        let mut similarities: Vec<(usize, f32)> = (0..num_windows)
            .into_par_iter()
            .map(|i| {
                let start = i * hop_size;
                let end = start + segment_len;
                let window = audio.slice(ndarray::s![.., start..end]);

                // Calculate similarity (negative distance)
                let distance = Self::matrix_distance(&segment, &window);
                (start, distance)
            })
            .collect();

        // Sort by distance (ascending = more similar)
        similarities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        similarities.truncate(self.k_neighbors);
        similarities
    }

    /// Calculate distance between two matrices
    fn matrix_distance(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }
}

/// Parallel batch processor for audio operations
pub struct BatchProcessor {
    #[allow(dead_code)]
    batch_size: usize,
    num_threads: Option<usize>,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            num_threads: None,
        }
    }

    /// Set number of threads for parallel processing
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Process items in parallel batches
    pub fn process<T, F, R>(&self, items: Vec<T>, f: F) -> Vec<R>
    where
        T: Send,
        F: Fn(T) -> R + Send + Sync,
        R: Send,
    {
        if let Some(threads) = self.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| items.into_par_iter().map(f).collect())
        } else {
            items.into_par_iter().map(f).collect()
        }
    }

    /// Process audio buffers in chunks
    pub fn process_chunks<F>(&self, data: &Array2<f32>, chunk_size: usize, f: F) -> Vec<Array2<f32>>
    where
        F: Fn(ArrayView2<f32>) -> Array2<f32> + Send + Sync,
    {
        let n_samples = data.ncols();
        let n_chunks = n_samples.div_ceil(chunk_size);

        (0..n_chunks)
            .into_par_iter()
            .map(|i| {
                let start = i * chunk_size;
                let end = (start + chunk_size).min(n_samples);
                let chunk = data.slice(ndarray::s![.., start..end]);
                f(chunk)
            })
            .collect()
    }
}

/// SIMD-optimized audio operations
pub struct SimdOps;

impl SimdOps {
    /// SIMD-optimized element-wise multiplication
    #[inline]
    pub fn multiply(a: &mut [f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x *= y);
    }

    /// SIMD-optimized element-wise addition
    #[inline]
    pub fn add(a: &mut [f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += y);
    }

    /// SIMD-optimized dot product
    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// SIMD-optimized RMS calculation
    #[inline]
    pub fn rms(data: &[f32]) -> f32 {
        let sum_squares: f32 = data.iter().map(|x| x * x).sum();
        (sum_squares / data.len() as f32).sqrt()
    }

    /// SIMD-optimized peak detection
    pub fn find_peaks(data: &[f32], threshold: f32) -> Vec<usize> {
        data.windows(3)
            .enumerate()
            .filter_map(|(i, window)| {
                if window[1] > threshold && window[1] > window[0] && window[1] > window[2] {
                    Some(i + 1)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_performance_hints() {
        let hints = PerformanceHints::new()
            .with_hint(PerformanceHint::Parallel)
            .with_hint(PerformanceHint::Vectorize);

        assert!(hints.has_hint(PerformanceHint::Parallel));
        assert!(hints.has_hint(PerformanceHint::Vectorize));
        assert!(!hints.has_hint(PerformanceHint::GpuAccelerated));
    }

    #[test]
    fn test_audio_knn() {
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0,
            ],
        )
        .unwrap();

        let query = Array2::from_shape_vec((1, 3), vec![2.5, 3.5, 4.5]).unwrap();

        let knn = AudioKNN::new(3);
        let neighbors = knn.find_neighbors(query.view(), data.view());

        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].len(), 3);
    }

    #[test]
    fn test_simd_ops() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];

        SimdOps::multiply(&mut a, &b);
        assert_eq!(a, vec![2.0, 4.0, 6.0, 8.0]);

        let rms = SimdOps::rms(&[1.0, 2.0, 3.0, 4.0]);
        assert_abs_diff_eq!(rms, 2.7386, epsilon = 0.001);
    }

    #[test]
    fn test_batch_processor() {
        let processor = BatchProcessor::new(10);
        let items: Vec<i32> = (0..100).collect();
        let results = processor.process(items, |x| x * 2);

        assert_eq!(results.len(), 100);
        assert_eq!(results[0], 0);
        assert_eq!(results[99], 198);
    }
}

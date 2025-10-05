//! Utility functions

use crate::error::Result;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Find audio files in directory
pub fn find_audio_files<P: AsRef<Path>>(dir: P, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let audio_extensions = ["wav", "mp3", "flac", "ogg", "m4a", "aac"];

    let walker = if recursive {
        WalkDir::new(dir)
    } else {
        WalkDir::new(dir).max_depth(1)
    };

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if let Some(ext_str) = ext.to_str() {
                    if audio_extensions.contains(&ext_str.to_lowercase().as_str()) {
                        files.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    Ok(files)
}

/// Format duration in seconds to human-readable string
pub fn format_duration(seconds: f64) -> String {
    let mins = (seconds / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let millis = ((seconds % 1.0) * 1000.0) as u32;
    format!("{mins:02}:{secs:02}.{millis:03}")
}

/// Calculate memory usage for audio buffer
pub fn estimate_memory_mb(samples: usize, channels: usize, sources: usize) -> f64 {
    let bytes = samples * channels * sources * std::mem::size_of::<f32>();
    bytes as f64 / (1024.0 * 1024.0)
}

/// Calculate optimal chunk size based on available memory
pub fn calculate_chunk_size(sample_rate: u32, channels: usize, target_memory_mb: f64) -> usize {
    let bytes_per_sample = channels * std::mem::size_of::<f32>();
    let target_bytes = target_memory_mb * 1024.0 * 1024.0;
    let samples = (target_bytes / bytes_per_sample as f64) as usize;

    // Round to nearest second
    let samples_per_second = sample_rate as usize;
    (samples / samples_per_second) * samples_per_second
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.0), "00:00.000");
        assert_eq!(format_duration(61.5), "01:01.500");
        assert_eq!(format_duration(3661.123), "61:01.123");
    }

    #[test]
    fn test_memory_estimation() {
        let mem = estimate_memory_mb(44100, 2, 4);
        assert!(mem > 0.0);
        assert!(mem < 2.0); // Should be around 1.4 MB
    }

    #[test]
    fn test_chunk_size_calculation() {
        let chunk = calculate_chunk_size(44100, 2, 100.0);
        assert!(chunk > 0);
        assert_eq!(chunk % 44100, 0); // Should be multiple of sample rate
    }
}

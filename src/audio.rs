//! Audio I/O and processing utilities

use crate::error::{CharonError, Result};
use hound::{WavSpec, WavWriter};
use ndarray::Array2;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::IntoSample;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Audio buffer holding multi-channel audio data
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Audio samples [channels, samples]
    pub data: Array2<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(data: Array2<f32>, sample_rate: u32) -> Self {
        Self { data, sample_rate }
    }

    /// Get number of channels
    pub fn channels(&self) -> usize {
        self.data.nrows()
    }

    /// Get number of samples per channel
    pub fn samples(&self) -> usize {
        self.data.ncols()
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f64 {
        self.samples() as f64 / self.sample_rate as f64
    }

    /// Convert to mono by averaging channels
    pub fn to_mono(&self) -> Array2<f32> {
        let mono = self.data.mean_axis(ndarray::Axis(0)).unwrap();
        mono.insert_axis(ndarray::Axis(0))
    }

    /// Resample to target sample rate
    pub fn resample(&self, target_rate: u32) -> Result<Self> {
        if self.sample_rate == target_rate {
            return Ok(self.clone());
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            target_rate as f64 / self.sample_rate as f64,
            2.0,
            params,
            self.samples(),
            self.channels(),
        )
        .map_err(|e| CharonError::Resampling(e.to_string()))?;

        // Convert to channel-major format for rubato
        let mut input_data: Vec<Vec<f32>> = Vec::new();
        for ch in 0..self.channels() {
            input_data.push(self.data.row(ch).to_vec());
        }

        let output_data = resampler
            .process(&input_data, None)
            .map_err(|e| CharonError::Resampling(e.to_string()))?;

        // Convert back to ndarray format
        let output_samples = output_data[0].len();
        let mut data = Array2::zeros((self.channels(), output_samples));
        for (ch, channel_data) in output_data.iter().enumerate() {
            for (i, &sample) in channel_data.iter().enumerate() {
                data[[ch, i]] = sample;
            }
        }

        Ok(AudioBuffer::new(data, target_rate))
    }

    /// Convert number of channels
    pub fn convert_channels(&self, target_channels: usize) -> Result<Self> {
        if self.channels() == target_channels {
            return Ok(self.clone());
        }

        let data = match (self.channels(), target_channels) {
            (1, 2) => {
                // Mono to stereo: duplicate channel
                let mono = self.data.row(0);
                ndarray::stack![ndarray::Axis(0), mono, mono]
            }
            (2, 1) => {
                // Stereo to mono: average channels
                self.to_mono()
            }
            (n, 1) if n > 1 => {
                // Multi-channel to mono: average all channels
                self.to_mono()
            }
            (n, m) if n > m => {
                // Downmix: take first m channels
                self.data.slice(ndarray::s![0..m, ..]).to_owned()
            }
            _ => {
                return Err(CharonError::Audio(format!(
                    "Unsupported channel conversion from {} to {}",
                    self.channels(),
                    target_channels
                )))
            }
        };

        Ok(AudioBuffer::new(data, self.sample_rate))
    }

    /// Normalize audio to [-1, 1] range
    pub fn normalize(&mut self) {
        let max_val = self.data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max_val > 0.0 {
            self.data /= max_val;
        }
    }

    /// Apply gain (in dB)
    pub fn apply_gain(&mut self, gain_db: f32) {
        let gain = 10.0f32.powf(gain_db / 20.0);
        self.data *= gain;
    }
}

/// Audio file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    Auto,
}

impl AudioFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|s| s.to_str()) {
            Some("wav") => AudioFormat::Wav,
            Some("mp3") => AudioFormat::Mp3,
            Some("flac") => AudioFormat::Flac,
            Some("ogg") => AudioFormat::Ogg,
            _ => AudioFormat::Auto,
        }
    }
}

/// Audio file reader/writer
pub struct AudioFile;

impl AudioFile {
    /// Read audio file with automatic format detection
    pub fn read<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let meta_opts = MetadataOptions::default();
        let fmt_opts = FormatOptions::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| CharonError::Audio(e.to_string()))?;

        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| CharonError::Audio("No supported audio track found".to_string()))?;

        let dec_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| CharonError::Audio(e.to_string()))?;

        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| CharonError::Audio("Sample rate not found".to_string()))?;

        let channels = track
            .codec_params
            .channels
            .ok_or_else(|| CharonError::Audio("Channel info not found".to_string()))?
            .count();

        let mut samples: Vec<Vec<f32>> = vec![Vec::new(); channels];

        while let Ok(packet) = format.next_packet() {
            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(_) => continue,
            };

            Self::copy_samples(&decoded, &mut samples);
        }

        // Convert to ndarray
        let num_samples = samples[0].len();
        let mut data = Array2::zeros((channels, num_samples));
        for (ch, channel_samples) in samples.iter().enumerate() {
            for (i, &sample) in channel_samples.iter().enumerate() {
                data[[ch, i]] = sample;
            }
        }

        Ok(AudioBuffer::new(data, sample_rate))
    }

    fn copy_samples(decoded: &AudioBufferRef, output: &mut [Vec<f32>]) {
        match decoded {
            AudioBufferRef::F32(buf) => {
                for (ch, out_ch) in output.iter_mut().enumerate().take(buf.spec().channels.count()) {
                    out_ch.extend_from_slice(buf.chan(ch));
                }
            }
            AudioBufferRef::S32(buf) => {
                for (ch, out_ch) in output.iter_mut().enumerate().take(buf.spec().channels.count()) {
                    out_ch.extend(buf.chan(ch).iter().map(|&s| IntoSample::<f32>::into_sample(s)));
                }
            }
            AudioBufferRef::S16(buf) => {
                for (ch, out_ch) in output.iter_mut().enumerate().take(buf.spec().channels.count()) {
                    out_ch.extend(buf.chan(ch).iter().map(|&s| IntoSample::<f32>::into_sample(s)));
                }
            }
            AudioBufferRef::U8(buf) => {
                for (ch, out_ch) in output.iter_mut().enumerate().take(buf.spec().channels.count()) {
                    out_ch.extend(buf.chan(ch).iter().map(|&s| IntoSample::<f32>::into_sample(s)));
                }
            }
            _ => {}
        }
    }

    /// Write audio buffer to WAV file
    pub fn write_wav<P: AsRef<Path>>(path: P, buffer: &AudioBuffer) -> Result<()> {
        let spec = WavSpec {
            channels: buffer.channels() as u16,
            sample_rate: buffer.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = WavWriter::create(path, spec)
            .map_err(|e| CharonError::Audio(e.to_string()))?;

        // Interleave samples
        for i in 0..buffer.samples() {
            for ch in 0..buffer.channels() {
                writer.write_sample(buffer.data[[ch, i]])
                    .map_err(|e| CharonError::Audio(e.to_string()))?;
            }
        }

        writer.finalize()
            .map_err(|e| CharonError::Audio(e.to_string()))?;
        Ok(())
    }

    /// Write audio buffer to file (format detected from extension)
    pub fn write<P: AsRef<Path>>(path: P, buffer: &AudioBuffer) -> Result<()> {
        let format = AudioFormat::from_path(path.as_ref());
        match format {
            AudioFormat::Wav | AudioFormat::Auto => Self::write_wav(path, buffer),
            _ => Err(CharonError::NotSupported(
                "Only WAV output is currently supported".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_audio_buffer_creation() {
        let data = Array2::zeros((2, 1000));
        let buffer = AudioBuffer::new(data, 44100);
        assert_eq!(buffer.channels(), 2);
        assert_eq!(buffer.samples(), 1000);
        assert_eq!(buffer.sample_rate, 44100);
    }

    #[test]
    fn test_duration_calculation() {
        let data = Array2::zeros((2, 44100));
        let buffer = AudioBuffer::new(data, 44100);
        assert_abs_diff_eq!(buffer.duration(), 1.0, epsilon = 0.001);
    }

    #[test]
    fn test_mono_conversion() {
        let mut data = Array2::zeros((2, 100));
        data.row_mut(0).fill(1.0);
        data.row_mut(1).fill(3.0);
        
        let buffer = AudioBuffer::new(data, 44100);
        let mono = buffer.to_mono();
        
        assert_eq!(mono.nrows(), 1);
        assert_abs_diff_eq!(mono[[0, 0]], 2.0, epsilon = 0.001);
    }
}

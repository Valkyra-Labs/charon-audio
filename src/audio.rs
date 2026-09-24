//! Audio I/O and processing utilities

use crate::error::{CharonError, Result};
use hound::{WavSpec, WavWriter};
use ndarray::Array2;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::path::Path;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Audio buffer holding multi-channel audio data
#[derive(Debug, Clone, PartialEq)]
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

    /// Resample to target sample rate.
    ///
    /// Windowed-sinc resampling (rubato). The output is time-aligned with
    /// the input (checked with impulses in the tests) and has
    /// `round(samples * target_rate / sample_rate)` samples.
    pub fn resample(&self, target_rate: u32) -> Result<Self> {
        if self.sample_rate == target_rate {
            return Ok(self.clone());
        }
        let channels = self.channels();
        let in_len = self.samples();
        let ratio = target_rate as f64 / self.sample_rate as f64;
        let out_len = (in_len as f64 * ratio).round() as usize;
        if in_len == 0 || out_len == 0 {
            return Ok(AudioBuffer::new(
                Array2::zeros((channels, out_len)),
                target_rate,
            ));
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        const CHUNK: usize = 4096;
        let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, CHUNK, channels)
            .map_err(|e| CharonError::Resampling(e.to_string()))?;
        // rubato 0.15 `SincFixedIn` output is already aligned with the input
        // (an impulse lands where expected without trimming), so
        // `output_delay()` is not subtracted. Only the tail needs flushing.
        let delay = 0;

        let input: Vec<&[f32]> = (0..channels)
            .map(|ch| {
                self.data
                    .row(ch)
                    .to_slice()
                    .expect("row-major contiguous audio data")
            })
            .collect();
        let mut output: Vec<Vec<f32>> = vec![Vec::with_capacity(out_len + delay); channels];

        let mut pos = 0;
        while pos + CHUNK <= in_len {
            let chunk: Vec<&[f32]> = input.iter().map(|ch| &ch[pos..pos + CHUNK]).collect();
            let out = resampler
                .process(&chunk, None)
                .map_err(|e| CharonError::Resampling(e.to_string()))?;
            for (dst, src) in output.iter_mut().zip(out) {
                dst.extend_from_slice(&src);
            }
            pos += CHUNK;
        }
        if pos < in_len {
            let chunk: Vec<&[f32]> = input.iter().map(|ch| &ch[pos..]).collect();
            let out = resampler
                .process_partial(Some(&chunk), None)
                .map_err(|e| CharonError::Resampling(e.to_string()))?;
            for (dst, src) in output.iter_mut().zip(out) {
                dst.extend_from_slice(&src);
            }
        }
        // Flush the filter so the last `delay` frames of real audio come out.
        while output[0].len() < out_len + delay {
            let out = resampler
                .process_partial::<Vec<f32>>(None, None)
                .map_err(|e| CharonError::Resampling(e.to_string()))?;
            if out[0].is_empty() {
                break;
            }
            for (dst, src) in output.iter_mut().zip(out) {
                dst.extend_from_slice(&src);
            }
        }

        let mut data = Array2::zeros((channels, out_len));
        for (ch, samples) in output.iter().enumerate() {
            let available = samples.len().saturating_sub(delay).min(out_len);
            data.row_mut(ch).slice_mut(ndarray::s![..available]).assign(
                &ndarray::ArrayView1::from(&samples[delay..delay + available]),
            );
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

    /// Peak absolute sample value
    pub fn peak(&self) -> f32 {
        self.data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
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
        match path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("wav") => AudioFormat::Wav,
            Some("mp3") => AudioFormat::Mp3,
            Some("flac") => AudioFormat::Flac,
            Some("ogg") => AudioFormat::Ogg,
            _ => AudioFormat::Auto,
        }
    }
}

/// Sample encoding for written files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BitDepth {
    Int16,
    Int24,
    /// 32-bit IEEE float. WAV only.
    #[default]
    Float32,
}

impl BitDepth {
    fn bits(self) -> u16 {
        match self {
            BitDepth::Int16 => 16,
            BitDepth::Int24 => 24,
            BitDepth::Float32 => 32,
        }
    }

    /// Convert a float sample to an integer of this depth, rounding to
    /// nearest and clipping at full scale.
    fn quantize(self, x: f32) -> i32 {
        let full_scale = 1i64 << (self.bits() - 1);
        let v = (x as f64 * full_scale as f64).round() as i64;
        v.clamp(-full_scale, full_scale - 1) as i32
    }
}

/// Audio file reader/writer
pub struct AudioFile;

impl AudioFile {
    /// Read audio file with automatic format detection.
    ///
    /// Samples are returned as decoded, without clipping: lossy codecs can
    /// produce values outside [-1, 1].
    pub fn read<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| CharonError::Audio(e.to_string()))?;

        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| CharonError::Audio("No audio track found".to_string()))?;
        let track_id = track.id;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| CharonError::Audio("No audio codec parameters".to_string()))?;
        let mut sample_rate = params.sample_rate;
        let mut channels = params.channels.as_ref().map(|c| c.count());

        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .map_err(|e| CharonError::Audio(e.to_string()))?;

        let mut samples: Vec<Vec<f32>> = Vec::new();
        let mut scratch: Vec<Vec<f32>> = Vec::new();
        loop {
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break
                }
                Err(e) => return Err(CharonError::Audio(e.to_string())),
            };
            if packet.track_id != track_id {
                continue;
            }
            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(SymphoniaError::DecodeError(msg)) => {
                    log::warn!("skipping undecodable packet: {msg}");
                    continue;
                }
                Err(e) => return Err(CharonError::Audio(e.to_string())),
            };

            let spec = decoded.spec();
            let planes = decoded.num_planes();
            let frames = decoded.frames();
            if samples.is_empty() {
                sample_rate.get_or_insert(spec.rate());
                channels.get_or_insert(planes);
                samples = vec![Vec::new(); planes];
                scratch = vec![Vec::new(); planes];
            }
            if planes != samples.len() {
                return Err(CharonError::Audio(format!(
                    "channel count changed mid-stream: {} -> {planes}",
                    samples.len()
                )));
            }
            for plane in &mut scratch {
                plane.resize(frames, 0.0);
            }
            decoded.copy_to_slice_planar::<f32, _>(&mut scratch);
            for (dst, src) in samples.iter_mut().zip(&scratch) {
                dst.extend_from_slice(src);
            }
        }

        let channels = channels
            .filter(|&c| c > 0)
            .ok_or_else(|| CharonError::Audio("No audio channels found".to_string()))?;
        let sample_rate =
            sample_rate.ok_or_else(|| CharonError::Audio("Sample rate not found".to_string()))?;
        if samples.is_empty() {
            samples = vec![Vec::new(); channels];
        }

        let num_samples = samples[0].len();
        let mut data = Array2::zeros((channels, num_samples));
        for (ch, channel_samples) in samples.iter().enumerate() {
            data.row_mut(ch)
                .assign(&ndarray::ArrayView1::from(channel_samples.as_slice()));
        }

        Ok(AudioBuffer::new(data, sample_rate))
    }

    /// Write audio buffer to a 32-bit float WAV file
    pub fn write_wav<P: AsRef<Path>>(path: P, buffer: &AudioBuffer) -> Result<()> {
        Self::write_wav_with_depth(path, buffer, BitDepth::Float32)
    }

    /// Write audio buffer to a WAV file with the given sample encoding.
    /// Integer depths round to nearest and clip at full scale.
    pub fn write_wav_with_depth<P: AsRef<Path>>(
        path: P,
        buffer: &AudioBuffer,
        depth: BitDepth,
    ) -> Result<()> {
        let spec = WavSpec {
            channels: buffer.channels() as u16,
            sample_rate: buffer.sample_rate,
            bits_per_sample: depth.bits(),
            sample_format: match depth {
                BitDepth::Float32 => hound::SampleFormat::Float,
                _ => hound::SampleFormat::Int,
            },
        };
        let wav_err = |e: hound::Error| CharonError::Audio(e.to_string());
        let mut writer = WavWriter::create(path, spec).map_err(wav_err)?;

        let frames = buffer.data.t();
        match depth {
            BitDepth::Float32 => {
                for &x in frames.iter() {
                    writer.write_sample(x).map_err(wav_err)?;
                }
            }
            _ => {
                for &x in frames.iter() {
                    writer.write_sample(depth.quantize(x)).map_err(wav_err)?;
                }
            }
        }
        writer.finalize().map_err(wav_err)
    }

    /// Write audio buffer to a FLAC file (16 or 24 bit)
    pub fn write_flac<P: AsRef<Path>>(
        path: P,
        buffer: &AudioBuffer,
        depth: BitDepth,
    ) -> Result<()> {
        use flacenc::component::BitRepr;
        use flacenc::error::Verify;

        if depth == BitDepth::Float32 {
            return Err(CharonError::NotSupported(
                "FLAC supports integer samples only; use BitDepth::Int16 or Int24".to_string(),
            ));
        }
        let samples: Vec<i32> = buffer.data.t().iter().map(|&x| depth.quantize(x)).collect();
        let config = flacenc::config::Encoder::default()
            .into_verified()
            .map_err(|(_, e)| CharonError::Audio(format!("FLAC config: {e}")))?;
        let source = flacenc::source::MemSource::from_samples(
            &samples,
            buffer.channels(),
            depth.bits() as usize,
            buffer.sample_rate as usize,
        );
        let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
            .map_err(|e| CharonError::Audio(format!("FLAC encode: {e}")))?;
        let mut sink = flacenc::bitsink::ByteSink::new();
        stream
            .write(&mut sink)
            .map_err(|e| CharonError::Audio(format!("FLAC write: {e}")))?;
        std::fs::write(path, sink.as_slice())?;
        Ok(())
    }

    /// Write audio buffer to file. The container comes from the extension:
    /// `.wav` (32-bit float) or `.flac` (24-bit).
    pub fn write<P: AsRef<Path>>(path: P, buffer: &AudioBuffer) -> Result<()> {
        match AudioFormat::from_path(path.as_ref()) {
            AudioFormat::Wav | AudioFormat::Auto => Self::write_wav(path, buffer),
            AudioFormat::Flac => Self::write_flac(path, buffer, BitDepth::Int24),
            _ => Err(CharonError::NotSupported(
                "Only WAV and FLAC output are supported".to_string(),
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

    #[test]
    fn test_quantize_rounds_and_clips() {
        assert_eq!(BitDepth::Int16.quantize(0.0), 0);
        assert_eq!(BitDepth::Int16.quantize(1.0), 32767);
        assert_eq!(BitDepth::Int16.quantize(-1.0), -32768);
        assert_eq!(BitDepth::Int16.quantize(1.5), 32767);
        assert_eq!(BitDepth::Int24.quantize(0.5), 4_194_304);
    }

    #[test]
    fn test_resample_is_time_aligned() {
        // Impulses near the start and the end must land at the resampled
        // position, and the output length must be the rounded ratio.
        for (from, to) in [
            (44100u32, 48000u32),
            (48000, 44100),
            (44100, 22050),
            (22050, 44100),
        ] {
            let n = 30000;
            let data = Array2::from_shape_fn((1, n), |(_, i)| {
                if i == 10000 || i == n - 1000 {
                    1.0
                } else {
                    0.0
                }
            });
            let out = AudioBuffer::new(data, from).resample(to).unwrap();
            let ratio = to as f64 / from as f64;
            assert_eq!(out.samples(), (n as f64 * ratio).round() as usize);
            for pos in [10000usize, n - 1000] {
                let expected = (pos as f64 * ratio).round() as usize;
                let window = out
                    .data
                    .slice(ndarray::s![0, expected - 200..expected + 200]);
                let (peak_idx, peak) =
                    window
                        .iter()
                        .enumerate()
                        .fold((0, 0.0f32), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
                let found = expected - 200 + peak_idx;
                // A fractional resampled position splits the impulse over two
                // samples, so allow one sample either way.
                assert!(
                    found.abs_diff(expected) <= 1,
                    "{from}->{to}: impulse at {pos} found at {found}, expected {expected}"
                );
                assert!(peak > 0.3, "{from}->{to}: impulse at {pos} has peak {peak}");
            }
        }
    }

    #[test]
    fn test_resample_sine_accuracy() {
        // 1 kHz sine, 44.1 -> 48 kHz, compared with the ideal signal away
        // from the edges. The bound is the measured error of the chosen
        // sinc parameters plus margin, not a spec.
        let n = 44100;
        let data = Array2::from_shape_fn((1, n), |(_, i)| {
            (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin()
        });
        let out = AudioBuffer::new(data, 44100).resample(48000).unwrap();
        let mut max_err = 0.0f32;
        for i in 2000..46000 {
            let ideal = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin();
            max_err = max_err.max((out.data[[0, i]] - ideal).abs());
        }
        assert!(max_err < 2e-2, "max error {max_err}");
    }

    #[test]
    fn test_resample_identity_rate() {
        let buffer = AudioBuffer::new(Array2::ones((2, 10)), 44100);
        assert_eq!(buffer.resample(44100).unwrap(), buffer);
    }
}

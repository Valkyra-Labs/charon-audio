//! Pipeline tests with a tiny identity ONNX model: every stem is a copy of
//! the model input, so the whole pipeline (normalization, segmentation,
//! context padding, overlap-add, shifts, tensor plumbing) must return the
//! input unchanged.
#![cfg(feature = "ort-backend")]

use charon_audio::{
    AudioBuffer, AudioFile, BitDepth, ExecutionProvider, ModelConfig, Separator, SeparatorConfig,
    StemFormat,
};
use ndarray::Array2;
use std::path::PathBuf;

const SEGMENT: usize = 1000;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Deterministic pseudo-random stereo signal in [-0.5, 0.5).
fn signal(samples: usize, seed: u64) -> Array2<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    Array2::from_shape_simple_fn((2, samples), || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    })
}

fn identity_separator(segment: Option<usize>, shifts: usize) -> Separator {
    let mut model = ModelConfig::htdemucs(fixture("identity_4stems.onnx"));
    model.segment_samples = segment;
    let mut config = SeparatorConfig {
        model,
        ..SeparatorConfig::default()
    }
    .with_shifts(shifts)
    .with_progress(false);
    config.process.segment_length = None;
    Separator::new(config).expect("identity model loads")
}

fn assert_identity(separator: &Separator, samples: usize) {
    let input = signal(samples, samples as u64);
    let audio = AudioBuffer::new(input.clone(), 44100);
    let stems = separator.separate(&audio).expect("separation succeeds");

    assert_eq!(stems.list(), ["drums", "bass", "other", "vocals"]);
    for name in stems.list() {
        let stem = stems.get(&name).unwrap();
        assert_eq!(stem.data.dim(), input.dim(), "stem {name}, len {samples}");
        let max_err = stem
            .data
            .iter()
            .zip(input.iter())
            .map(|(a, b)| {
                assert!(a.is_finite(), "non-finite sample in {name}, len {samples}");
                (a - b).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-5,
            "stem {name}, len {samples}: max error {max_err}"
        );
    }
}

#[test]
fn identity_model_round_trips_across_segment_boundaries() {
    let separator = identity_separator(Some(SEGMENT), 1);
    for samples in [1, SEGMENT - 1, SEGMENT, SEGMENT + 1, SEGMENT * 7 / 2] {
        assert_identity(&separator, samples);
    }
}

#[test]
fn identity_model_round_trips_with_shifts() {
    let separator = identity_separator(Some(SEGMENT), 3);
    for samples in [SEGMENT - 1, SEGMENT * 7 / 2] {
        assert_identity(&separator, samples);
    }
}

#[test]
fn identity_model_round_trips_without_segmentation() {
    let separator = identity_separator(None, 1);
    assert_identity(&separator, 2500);
}

#[test]
fn silent_input_stays_silent_and_finite() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let audio = AudioBuffer::new(Array2::zeros((2, 2500)), 44100);
    let stems = separator.separate(&audio).unwrap();
    for name in stems.list() {
        assert!(stems.get(&name).unwrap().data.iter().all(|&x| x == 0.0));
    }
}

#[test]
fn empty_input_is_an_error() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let audio = AudioBuffer::new(Array2::zeros((2, 0)), 44100);
    assert!(separator.separate(&audio).is_err());
}

#[test]
fn stems_are_written_as_wav_in_model_order() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let input = signal(3000, 7);
    let stems = separator
        .separate(&AudioBuffer::new(input.clone(), 44100))
        .unwrap();

    let dir = std::env::temp_dir().join(format!("charon_pipeline_{}", std::process::id()));
    stems.save_all(&dir).unwrap();
    for name in ["drums", "bass", "other", "vocals"] {
        let read = AudioFile::read(dir.join(format!("{name}.wav"))).unwrap();
        assert_eq!(read.sample_rate, 44100);
        assert_eq!(read.data.dim(), input.dim());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn wav_round_trip_is_exact() {
    let input = signal(4410, 42);
    let path = std::env::temp_dir().join(format!("charon_roundtrip_{}.wav", std::process::id()));
    AudioFile::write_wav(&path, &AudioBuffer::new(input.clone(), 48000)).unwrap();
    let read = AudioFile::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(read.sample_rate, 48000);
    assert_eq!(read.data, input);
}

#[test]
fn flac_and_int_wav_round_trip_through_symphonia() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let input = signal(3000, 11);
    let stems = separator
        .separate(&AudioBuffer::new(input.clone(), 44100))
        .unwrap();
    let dir = std::env::temp_dir().join(format!("charon_formats_{}", std::process::id()));

    for (format, ext, step) in [
        (StemFormat::Flac(BitDepth::Int24), "flac", 1.0 / 8_388_608.0),
        (StemFormat::Flac(BitDepth::Int16), "flac", 1.0 / 32_768.0),
        (StemFormat::Wav(BitDepth::Int24), "wav", 1.0 / 8_388_608.0),
    ] {
        let out = dir.join(format!("{format:?}"));
        stems.save_all_as(&out, format).unwrap();
        let read = AudioFile::read(out.join(format!("vocals.{ext}"))).unwrap();
        assert_eq!(read.sample_rate, 44100);
        assert_eq!(read.data.dim(), input.dim(), "{format:?}");
        let max_err = read
            .data
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Quantization error is at most half a step; the identity model
        // and the pipeline add float rounding on top.
        assert!(
            max_err <= step,
            "{format:?}: max error {max_err}, step {step}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The split contract with an identity model: the time branch returns the
/// mix and the spectral branch zeros, so stems must equal the input after
/// charon's STFT, inference, iSTFT and sum. Segment lengths cover partial
/// STFT frames.
#[test]
fn split_contract_identity_round_trips() {
    for segment in [8192usize, 12000] {
        let mut model = ModelConfig::htdemucs_split(fixture("identity_split.onnx"));
        model.segment_samples = Some(segment);
        model.onnx.execution_provider = ExecutionProvider::Cpu;
        let mut config = SeparatorConfig {
            model,
            ..SeparatorConfig::default()
        }
        .with_progress(false);
        config.process.segment_length = None;
        let separator = Separator::new(config).unwrap();
        for samples in [segment - 1, segment, segment * 5 / 2] {
            let input = signal(samples, samples as u64 + 3);
            let stems = separator
                .separate(&AudioBuffer::new(input.clone(), 44100))
                .unwrap();
            for name in stems.list() {
                let stem = stems.get(&name).unwrap();
                let max_err = stem
                    .data
                    .iter()
                    .zip(input.iter())
                    .map(|(a, b)| {
                        assert!(a.is_finite());
                        (a - b).abs()
                    })
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 1e-5,
                    "segment {segment}, {samples} samples, {name}: {max_err}"
                );
            }
        }
    }
}

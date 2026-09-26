//! Pipeline tests with a tiny identity ONNX model: every stem is a copy of
//! the model input, so the whole pipeline (normalization, segmentation,
//! context padding, overlap-add, shifts, tensor plumbing) must return the
//! input unchanged.
#![cfg(feature = "ort-backend")]

use charon_audio::{
    AudioBuffer, BufferSink, BufferSource, CancelToken, CharonError, Control, ExecutionProvider,
    ModelConfig, Progress, Region, RegionPlan, Separator, SeparatorConfig, REGION_OUTPUTS,
};
#[cfg(feature = "decode")]
use charon_audio::{AudioFile, BitDepth, StemFormat};
use ndarray::Array2;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
#[cfg(feature = "decode")]
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
#[cfg(feature = "decode")]
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
#[cfg(feature = "decode")]
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

/// Progress reports every window once, in order, and ends at the total
/// the job announced; the output is the same as without a control.
#[test]
fn progress_counts_every_window_and_does_not_change_output() {
    for (segment, shifts, samples) in [
        (Some(SEGMENT), 1, SEGMENT * 7 / 2),
        (Some(SEGMENT), 3, SEGMENT * 5),
        (None, 1, 2500),
    ] {
        let separator = identity_separator(segment, shifts);
        let audio = AudioBuffer::new(signal(samples, 7), 44100);
        let seen: Arc<Mutex<Vec<Progress>>> = Arc::default();
        let sink = seen.clone();
        let control = Control::new().with_progress(move |p| sink.lock().unwrap().push(p));

        let with = separator.separate_with(&audio, &control).unwrap();
        let without = separator.separate(&audio).unwrap();

        let seen = seen.lock().unwrap();
        assert!(!seen.is_empty(), "segment {segment:?}, shifts {shifts}");
        let total = seen[0].total;
        for (i, p) in seen.iter().enumerate() {
            assert_eq!(p.done, i + 1, "segment {segment:?}, shifts {shifts}");
            assert_eq!(p.total, total, "total must not change during a job");
        }
        assert_eq!(
            seen.last().unwrap().done,
            total,
            "job must end at its total"
        );
        assert!((seen.last().unwrap().fraction() - 1.0).abs() < f64::EPSILON);

        for name in without.list() {
            assert_eq!(
                with.get(&name).unwrap().data,
                without.get(&name).unwrap().data
            );
        }
    }
}

#[test]
fn cancelled_before_start_runs_no_window() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let audio = AudioBuffer::new(signal(SEGMENT * 4, 3), 44100);
    let token = CancelToken::new();
    token.cancel();
    let windows = Arc::new(Mutex::new(0usize));
    let counter = windows.clone();
    let control = Control::new()
        .with_cancel(token)
        .with_progress(move |_| *counter.lock().unwrap() += 1);

    let result = separator.separate_with(&audio, &control);
    assert!(
        matches!(result, Err(CharonError::Cancelled)),
        "{:?}",
        result.err()
    );
    assert_eq!(*windows.lock().unwrap(), 0);
}

#[test]
fn cancel_during_a_job_stops_after_the_current_window() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let audio = AudioBuffer::new(signal(SEGMENT * 10, 5), 44100);
    let token = CancelToken::new();
    let trigger = token.clone();
    let last = Arc::new(Mutex::new(None::<Progress>));
    let sink = last.clone();
    let control = Control::new().with_cancel(token).with_progress(move |p| {
        *sink.lock().unwrap() = Some(p);
        if p.done == 2 {
            trigger.cancel();
        }
    });

    let result = separator.separate_with(&audio, &control);
    assert!(
        matches!(result, Err(CharonError::Cancelled)),
        "{:?}",
        result.err()
    );
    let last = last.lock().unwrap().expect("some windows ran");
    assert_eq!(last.done, 2);
    assert!(last.total > 2);
}

fn streamed(
    separator: &Separator,
    audio: &AudioBuffer,
    control: &Control,
) -> Vec<(String, AudioBuffer)> {
    let mut sink = BufferSink::new();
    separator
        .separate_stream(&mut BufferSource::new(audio), &mut sink, control)
        .expect("streaming succeeds");
    sink.into_stems()
}

/// Streaming output equals whole-buffer output bit for bit, across window
/// boundaries and with a segment longer than the input.
#[test]
fn streaming_equals_whole_buffer() {
    for segment in [Some(SEGMENT), Some(SEGMENT * 3), None] {
        let separator = identity_separator(segment, 1);
        for samples in [
            1,
            SEGMENT - 1,
            SEGMENT,
            SEGMENT + 1,
            SEGMENT * 7 / 2,
            SEGMENT * 12 + 17,
        ] {
            let audio = AudioBuffer::new(signal(samples, 11 + samples as u64), 44100);
            let whole = separator.separate(&audio).unwrap();
            let stream = streamed(&separator, &audio, &Control::default());
            assert_eq!(stream.len(), whole.list().len());
            for (name, buffer) in stream {
                assert_eq!(
                    buffer.data,
                    whole.get(&name).unwrap().data,
                    "stem {name}, segment {segment:?}, {samples} samples"
                );
                assert_eq!(buffer.sample_rate, 44100);
            }
        }
    }
}

#[test]
fn streaming_reports_progress_and_cancels() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let audio = AudioBuffer::new(signal(SEGMENT * 10, 9), 44100);
    let token = CancelToken::new();
    let trigger = token.clone();
    let last = Arc::new(Mutex::new(None::<Progress>));
    let sink_progress = last.clone();
    let control = Control::new().with_cancel(token).with_progress(move |p| {
        *sink_progress.lock().unwrap() = Some(p);
        if p.done == 3 {
            trigger.cancel();
        }
    });
    let mut sink = BufferSink::new();
    let result = separator.separate_stream(&mut BufferSource::new(&audio), &mut sink, &control);
    assert!(
        matches!(result, Err(CharonError::Cancelled)),
        "{:?}",
        result.err()
    );
    assert_eq!(last.lock().unwrap().unwrap().done, 3);
}

#[test]
fn streaming_rejects_mismatched_input_and_shifts() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let wrong_rate = AudioBuffer::new(signal(SEGMENT * 2, 1), 48000);
    let mut sink = BufferSink::new();
    let result = separator.separate_stream(
        &mut BufferSource::new(&wrong_rate),
        &mut sink,
        &Control::default(),
    );
    assert!(
        matches!(result, Err(CharonError::InvalidConfig(_))),
        "{:?}",
        result.err()
    );

    let shifted = identity_separator(Some(SEGMENT), 2);
    let audio = AudioBuffer::new(signal(SEGMENT * 2, 1), 44100);
    let result = shifted.separate_stream(
        &mut BufferSource::new(&audio),
        &mut BufferSink::new(),
        &Control::default(),
    );
    assert!(
        matches!(result, Err(CharonError::NotSupported(_))),
        "{:?}",
        result.err()
    );
}

fn region_job(
    separator: &Separator,
    audio: &AudioBuffer,
    plan: &RegionPlan,
    control: &Control,
) -> (Array2<f32>, Array2<f32>) {
    let mut sink = BufferSink::new();
    separator
        .remove_in_regions(&mut BufferSource::new(audio), plan, &mut sink, control)
        .expect("region job succeeds");
    let mut stems = sink.into_stems().into_iter();
    let (result_name, result) = stems.next().unwrap();
    let (removed_name, removed) = stems.next().unwrap();
    assert_eq!(
        [result_name.as_str(), removed_name.as_str()],
        REGION_OUTPUTS
    );
    (result.data, removed.data)
}

/// With the identity model the "vocals" estimate is the input itself, so
/// full removal silences the region, and everything outside the regions
/// must be the input bit for bit, whatever the context and crossfade.
#[test]
fn region_removal_leaves_the_outside_bit_exact() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let len = SEGMENT * 9 + 123;
    let input = signal(len, 21);
    let audio = AudioBuffer::new(input.clone(), 44100);
    let regions = vec![
        Region {
            start: 6000,
            end: 7500,
            keep: 0.0,
        },
        Region {
            start: 500,
            end: 2600,
            keep: 0.0,
        },
        // Close to the previous one: their context spans overlap.
        Region {
            start: 2700,
            end: 3100,
            keep: 0.5,
        },
    ];
    for (context, crossfade) in [(0, 0), (300, 0), (700, 64), (5000, 200)] {
        let plan = RegionPlan {
            target: "vocals".into(),
            regions: regions.clone(),
            context,
            crossfade,
        };
        let (result, removed) = region_job(&separator, &audio, &plan, &Control::default());
        assert_eq!(result.dim(), input.dim());
        assert_eq!(removed.dim(), input.dim());
        for t in 0..len {
            let inside = regions.iter().find(|r| t >= r.start && t < r.end);
            for ch in 0..2 {
                let (x, y, z) = (input[[ch, t]], result[[ch, t]], removed[[ch, t]]);
                match inside {
                    None => {
                        assert_eq!(y.to_bits(), x.to_bits(), "outside, t={t}, ctx={context}");
                        assert_eq!(z, 0.0, "outside removed, t={t}");
                    }
                    Some(r) => {
                        assert!((y + z - x).abs() < 1e-5, "result + removed = mix, t={t}");
                        let far = t >= r.start + crossfade && t + crossfade < r.end;
                        if far {
                            let expect = x * (1.0 - r.keep);
                            assert!((z - expect).abs() < 1e-4, "t={t}: removed {z} vs {expect}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn region_crossfade_ramps_the_removal() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let len = SEGMENT * 4;
    // A constant signal makes the removed part equal to the fade itself.
    let audio = AudioBuffer::new(Array2::from_elem((2, len), 0.25), 44100);
    let plan = RegionPlan {
        target: "vocals".into(),
        regions: vec![Region {
            start: 1000,
            end: 3000,
            keep: 0.0,
        }],
        context: 400,
        crossfade: 100,
    };
    let (_, removed) = region_job(&separator, &audio, &plan, &Control::default());
    let row = removed.row(0);
    for t in 1000..1100 {
        assert!(row[t] < row[t + 1] + 1e-6, "rises at {t}");
        assert!(row[t] > 0.0);
    }
    assert!((row[2000] - 0.25).abs() < 1e-5);
    for t in 2900..2999 {
        assert!(row[t] + 1e-6 > row[t + 1], "falls at {t}");
    }
    assert_eq!(row[999], 0.0);
    assert_eq!(row[3000], 0.0);
}

#[test]
fn region_job_reports_one_progress_across_regions_and_validates() {
    let separator = identity_separator(Some(SEGMENT), 1);
    let audio = AudioBuffer::new(signal(SEGMENT * 8, 2), 44100);
    let seen: Arc<Mutex<Vec<Progress>>> = Arc::default();
    let sink = seen.clone();
    let control = Control::new().with_progress(move |p| sink.lock().unwrap().push(p));
    let plan = RegionPlan {
        target: "vocals".into(),
        regions: vec![
            Region {
                start: 100,
                end: 1500,
                keep: 0.0,
            },
            Region {
                start: 5000,
                end: 7000,
                keep: 0.0,
            },
        ],
        context: 200,
        crossfade: 10,
    };
    region_job(&separator, &audio, &plan, &control);
    let seen = seen.lock().unwrap();
    let total = seen[0].total;
    for (i, p) in seen.iter().enumerate() {
        assert_eq!(p.done, i + 1);
        assert_eq!(p.total, total);
    }
    assert_eq!(seen.last().unwrap().done, total);

    for (regions, target) in [
        (
            vec![Region {
                start: 10,
                end: 5,
                keep: 0.0,
            }],
            "vocals",
        ),
        (
            vec![Region {
                start: 0,
                end: SEGMENT * 9,
                keep: 0.0,
            }],
            "vocals",
        ),
        (
            vec![
                Region {
                    start: 0,
                    end: 500,
                    keep: 0.0,
                },
                Region {
                    start: 400,
                    end: 900,
                    keep: 0.0,
                },
            ],
            "vocals",
        ),
        (
            vec![Region {
                start: 0,
                end: 500,
                keep: 1.5,
            }],
            "vocals",
        ),
        (
            vec![Region {
                start: 0,
                end: 500,
                keep: 0.0,
            }],
            "music",
        ),
    ] {
        let plan = RegionPlan {
            target: target.into(),
            regions,
            context: 0,
            crossfade: 0,
        };
        let result = separator.remove_in_regions(
            &mut BufferSource::new(&audio),
            &plan,
            &mut BufferSink::new(),
            &Control::default(),
        );
        assert!(
            matches!(result, Err(CharonError::InvalidConfig(_))),
            "{:?}",
            result.err()
        );
    }
}

fn identity_spectral_separator() -> Separator {
    let mut model = ModelConfig::tiger_music(fixture("identity_spectral.onnx"));
    model.segment_samples = Some(SEGMENT);
    model.contract = charon_audio::ModelContract::Spectral {
        input: "spec".into(),
        output: "music_spec".into(),
        n_fft: 64,
        hop: 16,
    };
    model.onnx.execution_provider = ExecutionProvider::Cpu;
    let mut config = SeparatorConfig::tiger_music("unused.onnx").with_progress(false);
    config.model = model;
    Separator::new(config).expect("identity spectral model loads")
}

/// The spectral contract runs each channel through STFT, the model and
/// iSTFT; with an identity model every channel of any layout comes back
/// unchanged, across window boundaries, in batch and streaming.
#[test]
fn spectral_contract_identity_round_trips_any_channel_count() {
    let separator = identity_spectral_separator();
    for channels in [1usize, 2, 6] {
        for samples in [SEGMENT / 2, SEGMENT, SEGMENT * 3 + 77] {
            let mut data = Array2::zeros((channels, samples));
            for ch in 0..channels {
                data.row_mut(ch)
                    .assign(&signal(samples, (ch * 1000 + samples) as u64).row(0));
            }
            let audio = AudioBuffer::new(data.clone(), 44100);
            let stems = separator.separate(&audio).expect("separation succeeds");
            assert_eq!(stems.list(), ["music"]);
            let music = &stems.get("music").unwrap().data;
            assert_eq!(music.dim(), (channels, samples));
            let max_err = music
                .iter()
                .zip(data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 1e-5, "{channels} ch, {samples}: {max_err}");

            let stream = streamed(&separator, &audio, &Control::default());
            assert_eq!(stream[0].1.data, *music, "streaming equals batch");
        }
    }
}

/// A separator derived with other processing settings shares the loaded
/// model and behaves exactly like one built from scratch with them.
#[test]
fn derived_separator_equals_a_fresh_one_with_the_same_settings() {
    let base = identity_separator(Some(4096), 0);
    let mut process = base.process_config().clone();
    process.overlap = 0.5;
    let derived = base.with_process_config(process.clone());
    assert_eq!(derived.process_config().overlap, 0.5);
    assert_eq!(base.process_config().overlap, 0.25);

    let mut config = SeparatorConfig {
        model: base.model_config().clone(),
        ..SeparatorConfig::default()
    }
    .with_progress(false);
    config.process = process;
    let fresh = Separator::new(config).expect("identity model loads");

    let audio = AudioBuffer::new(signal(4096 * 3 + 101, 7), 44100);
    let a = derived.separate(&audio).expect("derived separates");
    let b = fresh.separate(&audio).expect("fresh separates");
    for name in b.list() {
        assert_eq!(
            a.get(&name).unwrap().data,
            b.get(&name).unwrap().data,
            "{name}"
        );
    }
}

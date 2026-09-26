//! Regression tests against PyTorch with real models: HTDemucs (in-graph and
//! split exports) and the TIGER-DnR music export.
//!
//! Ignored by default: needs the 316 MB model. Run with
//!
//! ```text
//! CHARON_HTDEMUCS_MODEL=/path/to/htdemucs.onnx cargo test --release -- --ignored
//! ```
//!
//! Model: https://huggingface.co/StemSplitio/htdemucs-onnx (`htdemucs.onnx`,
//! SHA-256 68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74).
//! Fixture: `tests/fixtures/make_htdemucs_fixture.py`.
#![cfg(all(feature = "ort-backend", feature = "decode"))]

use charon_audio::{AudioFile, BufferSink, BufferSource, Control, Separator, SeparatorConfig};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// Tolerance on 50 ms frame RMS, the Q1 parity criterion
/// (docs/parity/2026-09-24-htdemucs-q1.md). The measured max waveform
/// difference against PyTorch was 1.2e-4.
const RMS_TOLERANCE: f32 = 1e-3;

#[derive(Deserialize)]
struct Reference {
    frame_samples: usize,
    sources: Vec<String>,
    rms: HashMap<String, Vec<Vec<f32>>>,
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
#[ignore = "needs the HTDemucs ONNX model; set CHARON_HTDEMUCS_MODEL"]
fn htdemucs_matches_pytorch_reference() {
    let model = std::env::var("CHARON_HTDEMUCS_MODEL")
        .expect("set CHARON_HTDEMUCS_MODEL to the path of htdemucs.onnx");
    check_against_reference(SeparatorConfig::htdemucs(model));
}

/// Same check for the split-transform export (`tools/export/export_htdemucs.py`),
/// on whichever execution provider `Auto` picks.
#[test]
#[ignore = "needs the split HTDemucs export; set CHARON_HTDEMUCS_SPLIT_MODEL"]
fn htdemucs_split_matches_pytorch_reference() {
    let model = std::env::var("CHARON_HTDEMUCS_SPLIT_MODEL")
        .expect("set CHARON_HTDEMUCS_SPLIT_MODEL to the path of htdemucs_split.onnx");
    check_against_reference(SeparatorConfig::htdemucs_split(model));
}

/// The TIGER-DnR music export (`tools/export/export_tiger.py`) against its
/// PyTorch module on one centred window per channel
/// (`tests/fixtures/make_tiger_fixture.py`).
#[test]
#[ignore = "needs the TIGER music export; set CHARON_TIGER_MODEL"]
fn tiger_music_matches_pytorch_reference() {
    let model = std::env::var("CHARON_TIGER_MODEL")
        .expect("set CHARON_TIGER_MODEL to the path of tiger_music.onnx");
    let mut config = SeparatorConfig::tiger_music(model);
    config.model.onnx.execution_provider = charon_audio::ExecutionProvider::Cpu;
    check_against(config, "synth_9s_tiger_rms.json");
}

/// Streaming separation of a real model equals whole-buffer separation.
/// The fixture is resampled to the model rate first, as a streaming host
/// would while decoding.
#[test]
#[ignore = "needs the split HTDemucs export; set CHARON_HTDEMUCS_SPLIT_MODEL"]
fn htdemucs_split_streaming_equals_whole_buffer() {
    let model = std::env::var("CHARON_HTDEMUCS_SPLIT_MODEL")
        .expect("set CHARON_HTDEMUCS_SPLIT_MODEL to the path of htdemucs_split.onnx");
    let separator = Separator::new(SeparatorConfig::htdemucs_split(model)).unwrap();
    let audio = AudioFile::read(fixture("synth_9s.flac")).unwrap();
    let audio = audio
        .resample(separator.model_config().sample_rate)
        .unwrap()
        .convert_channels(separator.model_config().channels)
        .unwrap();

    let whole = separator.separate(&audio).unwrap();
    let mut sink = BufferSink::new();
    separator
        .separate_stream(
            &mut BufferSource::new(&audio),
            &mut sink,
            &Control::default(),
        )
        .unwrap();
    for (name, stream) in sink.into_stems() {
        let whole = whole.get(&name).unwrap();
        let max_diff = stream
            .data
            .iter()
            .zip(whole.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "{name}: max |stream - whole| = {max_diff:e} ({})",
            separator.provider()
        );
        assert!(max_diff <= 1e-6, "{name}: {max_diff}");
    }
}

fn check_against_reference(config: SeparatorConfig) {
    check_against(config, "synth_9s_htdemucs_rms.json");
}

fn check_against(config: SeparatorConfig, reference_file: &str) {
    let reference: Reference =
        serde_json::from_str(&std::fs::read_to_string(fixture(reference_file)).unwrap()).unwrap();

    let audio = AudioFile::read(fixture("synth_9s.flac")).unwrap();
    assert_eq!((audio.channels(), audio.samples()), (2, 9 * 44100));

    let separator = Separator::new(config.with_progress(false)).unwrap();
    let stems = separator.separate(&audio).unwrap();
    assert_eq!(stems.list(), reference.sources);

    let frame = reference.frame_samples;
    let mut worst = (0.0f32, String::new());
    for name in &reference.sources {
        let stem = stems.get(name).unwrap();
        for (ch, expected) in reference.rms[name].iter().enumerate() {
            let row = stem.data.row(ch);
            for (i, &want) in expected.iter().enumerate() {
                let chunk = row.slice(ndarray::s![i * frame..(i + 1) * frame]);
                let got = (chunk.iter().map(|x| x * x).sum::<f32>() / frame as f32).sqrt();
                assert!(got.is_finite(), "{name} ch{ch} frame {i}: non-finite");
                let diff = (got - want).abs();
                if diff > worst.0 {
                    worst = (diff, format!("{name} ch{ch} frame {i}: {got} vs {want}"));
                }
            }
        }
    }
    println!("max frame-RMS difference: {:.2e} ({})", worst.0, worst.1);
    assert!(worst.0 <= RMS_TOLERANCE, "{}", worst.1);
}

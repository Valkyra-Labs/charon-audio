//! Regression test against PyTorch Demucs with the real HTDemucs ONNX model.
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
#![cfg(feature = "ort-backend")]

use charon_audio::{AudioFile, Separator, SeparatorConfig};
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
    let reference: Reference = serde_json::from_str(
        &std::fs::read_to_string(fixture("synth_9s_htdemucs_rms.json")).unwrap(),
    )
    .unwrap();

    let audio = AudioFile::read(fixture("synth_9s.flac")).unwrap();
    assert_eq!((audio.channels(), audio.samples()), (2, 9 * 44100));

    let separator = Separator::new(SeparatorConfig::htdemucs(model).with_progress(false)).unwrap();
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

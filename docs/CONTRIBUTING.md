# Contributing

## Ground rules

- Every claim in the README is backed by a test in this repository or a
  recorded measurement (`docs/MEASUREMENTS.md`). A change that alters a
  number changes the record too.
- One variable per measurement. A commit that changes two things gives
  an unattributable result.
- Numbers from papers, model cards or vendors are hypotheses until
  reproduced here on a stated load shape.
- Nothing is "done" because it compiles. The acceptance gate is below.

## Acceptance gate

Run all of this before opening a pull request; CI runs the same set.

```bash
cargo fmt --all -- --check
```

```bash
cargo clippy --all-targets -- -D warnings
```

```bash
cargo clippy --all-targets --no-default-features -- -D warnings
```

```bash
cargo test
```

```bash
cargo test --no-default-features
```

```bash
cargo clippy --all-targets --no-default-features --features ort-backend -- -D warnings && cargo test --no-default-features --features ort-backend
```

```bash
cargo clippy --all-targets --no-default-features --features ort-backend,decode -- -D warnings && cargo test --no-default-features --features ort-backend,decode
```

On macOS also:

```bash
cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features
```

`--all-features` includes `coreml` and `ep-experimental`, which need
ONNX Runtime builds that exist for macOS only; on Linux use
`--features realtime` (needs `libasound2-dev`).

With a model present, the real-model regression tests:

```bash
CHARON_HTDEMUCS_MODEL=/path/htdemucs.onnx CHARON_HTDEMUCS_SPLIT_MODEL=/path/htdemucs_split_coreml.onnx cargo test --release --features coreml -- --ignored
```

## Toolchain

Rust 1.89 or later (`rust-version` in `Cargo.toml`, checked in CI with
`--locked`). `Cargo.lock` is committed: `ort` is a release candidate
whose API changes between RCs, so the lockfile is part of the build.
Bump `ort` deliberately and re-run the gate.

Python is needed only for the export and parity tools
(`tools/export/`, `tools/parity/`): demucs 4.1.0, torch, onnx,
onnxruntime, soundfile, numpy. Pin demucs; other packages that depend on
`demucs` have replaced it with an older version before.

## Layout

- `src/audio.rs`: decoding (Symphonia, features `decode` and `aac`),
  resampling (rubato), WAV/FLAC writing.
- `src/control.rs`: progress reporting and cancellation.
- `src/stft.rs`: `HTDemucs._spec`/`_ispec` with realfft, fixture-tested
  against `torch.stft`.
- `src/models.rs`: ONNX Runtime session, `ModelContract`, `OnnxOptions`,
  execution providers.
- `src/processor.rs`: segmentation, overlap-add, shifts, normalization
  (follows demucs 4.1.0 `apply_model`).
- `src/separator.rs`: `Separator`, `SeparatorConfig`, `Stems`.
- `src/bin/charon.rs`: the CLI and the resident server.
- `tests/pipeline.rs`: whole pipeline through tiny identity ONNX models
  (`tests/fixtures/make_identity_*.py`).
- `tests/htdemucs.rs`: ignored tests against PyTorch envelopes.
- `tools/export/`: the ONNX export; `tools/parity/`: measurement scripts.

## Adding a model

1. Define its tensor contract (`ModelContract`) and a preset on
   `ModelConfig`/`SeparatorConfig`.
2. Add an identity ONNX fixture for the contract and a pipeline test.
3. Measure parity against the model's reference implementation on real
   input (max difference at most 1e-3) and its quality on the MUSDB18
   previews with `tools/parity/musdb_eval.py`. Record both.
4. Add the file's hash and license position to `docs/MODELS.md` and
   `models/manifest.json`.

## Commits and pull requests

Commit messages: `<type>(<crate>): <short description>` with type in
feat, fix, refactor, test, docs, chore. A pull request states what was
measured, on what, and what the numbers were before and after.

## Licensing of contributions

Charon is licensed under either of Apache-2.0 or MIT, at your option.
Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

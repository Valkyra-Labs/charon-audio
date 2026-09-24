# Changelog

## 0.1.1 (unreleased)

### Fixed
- The default build compiles again. `Cargo.lock` is now committed, and
  `ort` is pinned to `=2.0.0-rc.13`. The old caret requirement on
  `2.0.0-rc.9` floated to newer, API-incompatible release candidates.
- `cargo test --no-default-features` compiles. Tests and examples that
  need `SeparatorConfig::onnx` now require the `ort-backend` feature.

### Added
- Real ONNX inference. `OnnxModel::infer` runs the session and maps the
  `[1, sources, channels, samples]` output onto the stems. Tensor names
  and a fixed segment length are set per model (`ModelConfig::input_name`,
  `output_name`, `segment_samples`).
- `ModelConfig::htdemucs` / `SeparatorConfig::htdemucs` for the 4-stem
  HTDemucs ONNX export (`StemSplitio/htdemucs-onnx`, `htdemucs.onnx`).
- Tests (`tests/pipeline.rs`) that run the full pipeline through a
  174-byte identity ONNX model. They cover segment boundaries, time
  shifts, silence, empty input, stem order and WAV round-trip.
- Ignored real-model test (`tests/htdemucs.rs`). It compares HTDemucs
  output with PyTorch demucs 4.1.0 reference envelopes on a 9 s FLAC
  fixture. Run it with `CHARON_HTDEMUCS_MODEL=... cargo test -- --ignored`.
- `tools/parity/`: parity harness against PyTorch demucs and an
  onnxruntime port.
- `Stems::from_ordered`. `Stems::list` and `save_all` follow model
  output order.

### Changed
- `cpal` and the `realtime` module are behind the new `realtime` feature,
  which is off by default.
- `anyhow` and `env_logger` moved to dev-dependencies. `num-traits`,
  `serde_json` and `candle-transformers` were removed as unused.
- The processing pipeline now follows Demucs 4.1.0 `apply_model`:
  - triangular overlap-add weights;
  - the last segment is centred with real context around it;
  - input is normalized by the mono reference mean and unbiased std;
  - time shifts zero-pad instead of rolling, with deterministic offsets.
  This fixes a NaN on the first output sample, which came from a zero
  overlap-add weight.
- Segments run sequentially, and ONNX Runtime uses all cores for each
  segment. `ProcessConfig::num_jobs` was removed.
- `ort` is built without its `ndarray` and `tracing` features, which
  removes the second copy of ndarray from the build.
- Declared `rust-version = "1.89"`, measured with the committed lockfile.
  Rust 1.86 fails. 1.87 and 1.88 were not tested.

### Removed (tombstones)
- `candle-backend` feature, `ModelBackend::Candle`, `CandleModel`,
  `SeparatorConfig::candle`. The backend was an identity placeholder,
  and candle-core 0.6 no longer builds. It will be reintroduced only
  together with a real model.
- `wasm` module and `WasmSeparator`. The code did not compile for
  wasm32, and no backend could run there.
- `cuda`, `tensorrt`, `metal`, `accelerate` features. They only forwarded
  build flags; no execution provider was ever registered at runtime.
  Execution providers return in a later release, each with a measurement.

### Known limitations
- Only HTDemucs is supported. Other ONNX models need their own contract.
- Peak memory is 3.8-5.4 GB for 5-31 s inputs on the CPU execution
  provider, and has not yet been profiled.

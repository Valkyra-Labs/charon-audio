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
- `OnnxOptions` on `ModelConfig`: intra-op threads, graph optimization
  level, memory pattern, CPU arena, disabled graph transformers.
  `OnnxOptions::low_memory()` (the HTDemucs preset default) and
  `OnnxOptions::max_speed()` (ONNX Runtime defaults). Measured on a
  193 s track: peak RSS 5.58 GB -> 2.12 GB for an 11% throughput cost
  (`docs/parity/2026-09-24-memory.md`).
- Output formats: `StemFormat` with WAV 16/24-bit int and 32-bit float,
  and FLAC 16/24-bit (`flacenc`). `Stems::save_all_as`,
  `AudioFile::write_wav_with_depth`, `AudioFile::write_flac`.
- `examples/profile.rs`: stage timings and peak RSS. `separate` example
  takes `--shifts`, `--format`, `--max-speed`.
- CI workflow: fmt, clippy and tests over the feature matrix on Linux and
  macOS, MSRV check, docs, and a weekly run of the real-model test.
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
- Symphonia 0.5 -> 0.6 with explicit codec/container features. Its MP3
  decoder no longer hard-clips to [-1, 1]. Decoder errors are logged, not
  silently skipped, and carry the failing stage.
- Resampling processes in 4096-frame chunks with a flushed tail, and is
  verified time-aligned with impulse tests at four rate pairs.
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
- CPU execution provider only. The ONNX Runtime 1.28 CoreML provider
  fails on the HTDemucs export in every configuration tried
  (`docs/parity/2026-09-24-memory.md`); CUDA has not been measured
  (no hardware).
- Peak RSS with the default low-memory preset is about 2.1 GB for any
  input length; about 0.7 GB of it is the loaded session.
- Segments run one at a time; ONNX Runtime uses all cores within a
  segment.

# Changelog

## 0.1.1 (unreleased)

### Fixed
- The default build compiles again. `Cargo.lock` is now committed, and
  `ort` is pinned to `=2.0.0-rc.13`. The old caret requirement on
  `2.0.0-rc.9` floated to newer, API-incompatible release candidates.
- `cargo test --no-default-features` compiles. Tests and examples that
  need `SeparatorConfig::onnx` now require the `ort-backend` feature.

### Added
- Split-transform HTDemucs path: `tools/export/export_htdemucs.py`
  exports the network with STFT/iSTFT outside the graph (185 MB,
  parity-checked against PyTorch); `src/stft.rs` implements
  `HTDemucs._spec` / `_ispec` with realfft, verified against `torch.stft`
  fixtures; `ModelContract::DemucsSplit`,
  `SeparatorConfig::htdemucs_split`.
- `coreml` feature: Apple CoreML execution provider for the split export
  (MLProgram, static shapes, compiled model cached in a directory keyed
  by the model file's SHA-256, since ONNX Runtime keys its cache by path
  and loads a stale compiled model otherwise). `ExecutionProvider::Auto`
  (default) uses CoreML when the feature is on and the session builds,
  else CPU. The CoreML-target export (`--target coreml`) tiles the time
  branch's convolutions so the whole graph runs as one CoreML partition.
  Measured on a 193 s track (M4 Pro): separation 5.6 s on CoreML vs
  16.6 s on CPU with the split exports and 22.8 s with the in-graph
  export; end to end 13.4 s vs 17.1 s vs 26.4 s
  (`docs/MEASUREMENTS.md`). CoreML output agrees
  with PyTorch to 2e-6.
- `charon` binary: `separate`, `serve` (resident server over a Unix
  socket, JSON lines), `ping`, `stop`. With the server warm a 193 s track
  takes 6.3 s on CoreML end to end, against 13.4 s cold (the 7 s CoreML
  load is paid once per server).
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
  (`docs/MEASUREMENTS.md`).
- Output formats: `StemFormat` with WAV 16/24-bit int and 32-bit float,
  and FLAC 16/24-bit (`flacenc`). `Stems::save_all_as`,
  `AudioFile::write_wav_with_depth`, `AudioFile::write_flac`.
- `examples/profile.rs`: stage timings and peak RSS. `separate` example
  takes `--shifts`, `--format`, `--max-speed`.
- CI workflow: fmt, clippy and tests over the feature matrix on Linux and
  macOS, MSRV check, docs, and a weekly run of the real-model test.
- `docs/MEASUREMENTS.md`: parity, MUSDB18 preview quality, memory and
  provider experiments, head-to-head against PyTorch Demucs (CPU, MPS,
  cold and resident), stem-splitter-core and demucs-rs on one machine,
  and shift ensemble gain. `docs/MODELS.md` and `models/manifest.json`:
  artifacts, hashes, hosting steps. `docs/IMPLEMENTATION.md`,
  `docs/CONTRIBUTING.md`, `docs/SUMMARY.md` rewritten to describe the
  code as it is.
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
- The split CPU preset uses graph optimization level `Extended`: level
  3's layout transforms cost 4% on this graph on Apple Silicon.
- Declared `rust-version = "1.89"`, measured with the committed lockfile.
  Rust 1.86 fails. 1.87 and 1.88 were not tested.

### Deprecated
- `performance` module (`SimdOps`, `AudioKNN`, `BatchProcessor`,
  `PerformanceHint(s)`): unused by the pipeline, kept for 0.1.0 API
  compatibility, removed in 0.2.
- `ModelZoo` now lists one real entry (HTDemucs, with URL and SHA-256)
  instead of three placeholders with `example.com` URLs. It still does
  not download.

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
- GPU: CoreML only (macOS, `coreml` feature, split export). The in-graph
  STFT export fails on the ORT 1.28 CoreML provider in every
  configuration tried (`docs/MEASUREMENTS.md`). CUDA has not
  been measured (no hardware). WebGPU runs but is slower than CPU.
- The split exports are not hosted yet; produce them with
  `tools/export/export_htdemucs.py --target cpu|coreml` (needs PyTorch
  and demucs 4.1.0; the CPU export is byte-for-byte reproducible, the
  CoreML export structurally; hashes of the measured artifacts recorded).
- Peak RSS: in-graph export with the low-memory preset about 2.1 GB;
  split export on CPU about 3.4 GB, on CoreML about 2.6 GB (ONNX
  Runtime activation memory; see the GPU record). Loading a cached
  CoreML model takes about 7 s per process; no ONNX Runtime option
  changes that.
- Segments run one at a time; ONNX Runtime uses all cores within a
  segment.

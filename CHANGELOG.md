# Changelog

## 0.1.1 (unreleased)

### Fixed
- The default build compiles again. `Cargo.lock` is now committed, and
  `ort` is pinned to `=2.0.0-rc.13`. The old caret requirement on
  `2.0.0-rc.9` floated to newer, API-incompatible release candidates.
- `cargo test --no-default-features` compiles. Tests and examples that
  need `SeparatorConfig::onnx` now require the `ort-backend` feature.

### Changed
- `cpal` and the `realtime` module are behind the new `realtime` feature,
  which is off by default.
- `anyhow` and `env_logger` moved to dev-dependencies. `num-traits`,
  `serde_json` and `candle-transformers` were removed as unused.
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
- ONNX inference is still a placeholder: `OnnxModel::infer` returns copies
  of the input. Real inference (HTDemucs) is the next work item.

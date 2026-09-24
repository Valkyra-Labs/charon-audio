# charon-audio: audit report and development plan

Date: 2026-09-24. Audited commit: `ba58451` (main, clean tree).
Host: Apple M4 Pro, 24 GB, macOS 26.6 (Darwin 25.6.0).
Toolchain: rustc 1.93.0-nightly (c871d09d1 2025-11-24), the host default.

## 1. Verdict

The crate does not perform source separation, and on a fresh checkout it
does not compile.

- `main` fails to build with default features (section 2.1).
- When the build is repaired by pinning dependencies, both inference
  backends are identity placeholders. Every "stem" is a copy of the input
  mix (section 2.3, measured).
- The pipeline around the placeholder has real defects: a NaN on the
  first sample, stems that sum to N times the mix, and a normalization
  scheme that no pretrained model expects.
- Real in the repository: audio decoding (Symphonia), WAV writing
  (hound), resampling (Rubato), a segmentation/overlap-add skeleton,
  and file discovery. That is usable plumbing, but it is not the product.

Until at least the "minimum programme" (section 5) lands, the project
should not be cited as a working separation library.

## 2. What was run, and what happened

### 2.1 Build and tests at HEAD (as a user gets it)

| Command | Result |
|---|---|
| `cargo fmt --check` | pass |
| `cargo build --all-targets` (default = `ort-backend`) | **FAIL**: 2x E0277 in `src/models.rs:77-78` |
| `cargo test` | **FAIL**: does not compile |
| `cargo test --no-default-features` | **FAIL**: `SeparatorConfig::onnx` is cfg-gated, but the test at `src/separator.rs:275` is not |
| `cargo clippy --no-default-features --all-targets -D warnings` | **FAIL**: 4 errors (1 compile error, 3 unused warnings) |
| `cargo build --no-default-features --examples` | **FAIL**: `separate` and `batch` call the gated `onnx()` |
| `cargo build --no-default-features --features candle-backend` | **FAIL**: candle-core 0.6.0 does not build against the current `rand`/`half` (bf16 `SampleUniform`) |
| `cargo check --target wasm32-unknown-unknown --no-default-features` (pinned copy) | **FAIL**: 3 errors in `src/wasm.rs` (missing `show_progress`, private field `config`, `AudioBuffer: !Serialize`) |
| README "Rust 1.70 or later" | **FALSE**: `cargo +1.82.0 check` cannot parse `coreaudio-sys` (edition 2024). The real MSRV is not established. |

Root cause of the default-build failure: `Cargo.lock` is in `.gitignore`,
and `ort = "2.0.0-rc.9"` is a caret requirement. It currently resolves to
`2.0.0-rc.13`, which changed `SessionBuilder` error types
(`ort::Error<SessionBuilder>`). Pinning `ort = "=2.0.0-rc.9"` alone is not
enough: `ort-sys` also floats to rc.13, and its build script then demands
a TLS feature. Both must be pinned.

The removed CI workflow (commit `a45c446`) ran
`cargo test --no-default-features --features ort-backend` on
ubuntu/macos/windows. Likely failure causes, none verified in CI:
missing `libasound2-dev` for `cpal` on Linux, and the same `ort` drift.
CI was deleted rather than fixed, and the README badge was removed.

### 2.2 Build and tests with `ort`/`ort-sys` pinned to `=2.0.0-rc.9` (scratch copy, not committed)

| Command | Result |
|---|---|
| `cargo test` | pass: 19 unit tests + 1 doctest (compile-only) |
| `cargo clippy --all-targets -- -D warnings` | pass |
| `cargo build --examples` | pass |

The 19 tests cover config defaults, a fade window, `format_duration`,
SIMD helper arithmetic and KNN output shapes. **No test runs a model.
No test checks separation. No test reads or writes an audio file.**

### 2.3 End-to-end run (pinned copy, release)

Input: 15 s, 44.1 kHz, stereo, 16-bit, 440 Hz sine at -6 dB (sox).
Model: a valid 66-byte ONNX graph containing a single `Identity` node.
Command: `cargo run --release --example separate -- mix.wav out/ dummy.onnx`.

| Observation | Value |
|---|---|
| Exit status, stems written | success, 4 x 5.29 MB WAV (drums, bass, vocals, other) |
| max \|stem - mix\| over finite frames, each stem | 5.96e-08 (float rounding: each stem equals the input) |
| NaN frames per stem | 1 (frame 0) |
| RMS(sum of stems) / RMS(mix) | 4.0 |

Any valid `.onnx` file "works", because `OnnxModel::infer`
(`src/models.rs:85-97`) never calls the session. It returns
`vec![input.clone(); num_sources]`. `CandleModel::infer` does the same
thing through a tensor round trip, even when weights are loaded.

### 2.4 Benchmarks (pinned copy, `cargo bench -- --warm-up-time 1 --measurement-time 3`)

Stamp: `ba58451` + ort/ort-sys pin, default features, bench profile.

| Group | n | median |
|---|---|---|
| audio_resampling 44.1 -> 48 kHz, 2 ch | 1k / 10k / 100k samples | 775 us / 1.51 ms / 8.80 ms |
| simd multiply | 100 / 1k / 10k | 30.2 ns / 99.9 ns / 1.34 us |
| simd rms | 100 / 1k / 10k | 18.4 ns / 427 ns / 4.97 us |
| knn_search | 100 / 1k / 5k | 16.5 / 16.7 / 16.8 us |
| audio_normalization, 2 ch | 1k / 10k / 100k | 377 ns / 4.28 us / 38.1 us |

How to read these:
- `knn_search` is flat because it measures nothing that scales.
  `find_neighbors` treats rows as points, so a `(2, n)` array is 2
  points, and `zip` truncates every distance to the 100-column query. The
  benchmark computes 2 distances of 100 floats each, regardless of `n`.
- "SIMD" functions are plain iterator loops. Any vectorization is the
  compiler's auto-vectorization, not code in this crate. `rms` at 10k is
  about 4x slower than `multiply` because an f32 sum reduction does not
  auto-vectorize without reassociation.
- No benchmark measures separation, model inference, or file I/O. The
  README performance table (2.1 s vs 12.5 s, 6x, 7x, 15x) has no source
  in the repository. It cannot have been measured, because no separation
  code exists.

## 3. README claims vs repository

Legend: **ABSENT** = no code; **PLACEHOLDER** = the API exists and the body
is a stub or identity; **BROKEN** = code exists and does not compile or
misbehaves; **PARTIAL** = works with caveats; **OK** = verified.

| README claim | Status | Evidence |
|---|---|---|
| "state-of-the-art ML inference" | **ABSENT** | No model runs. There is no measurement of any kind. |
| "imbalanced learning performance optimizations" / "patterns from rust-imbalanced-learn" | **ABSENT** | Not a dependency. The "patterns" are a naive O(n*m) KNN, iterator loops, and a `Vec<enum>` of hints. Imbalanced learning (SMOTE etc.) has no role in source separation inference. `docs/SUMMARY.md` addresses the reader as "your rust-imbalanced-learn library": pasted chat output. |
| Pure Rust, no Python | **PARTIAL** | True for the code. The `ort` backend downloads the native C++ ONNX Runtime library at build time. |
| ONNX Runtime backend | **PLACEHOLDER** + **BROKEN** | `src/models.rs:85-97`: returns input copies. Does not compile at HEAD. |
| HuggingFace Candle backend | **PLACEHOLDER** + **BROKEN** | `src/models.rs:143-177`: identity. candle 0.6 does not build. `candle-transformers` is never used. |
| Decode MP3/FLAC/OGG/WAV (Symphonia) | **PARTIAL** | Works for f32/s32/s16/u8 sample formats. Other formats (s24, f64, ...) are dropped silently (`_ => {}`, `src/audio.rs:291`), and so are decode errors (`Err(_) => continue`). Panics on zero channels (`samples[0]`). No file-based test exists. |
| High-quality resampling (Rubato) | **PARTIAL** | Works one-shot on the whole file. Filter delay is not compensated, so output length and alignment are slightly off. Memory scales with file length. |
| Real-time processing with CPAL | **PLACEHOLDER** | `src/realtime.rs` runs model inference inside the audio input callback while holding a `Mutex`: not real-time safe. Output is never sent to a device. `_processor` is unused. The example calls itself "conceptual" and does not use `RealtimeSeparator`. |
| CUDA / TensorRT / Metal / Accelerate | **ABSENT** | The features only forward flags to `ort`/`candle`. No execution provider is ever registered on the ONNX session. Metal/Accelerate apply to candle only, which does not build. CoreML is not mentioned. |
| SIMD-optimized operations | **ABSENT** | Scalar iterator loops (section 2.4). |
| KNN-based audio similarity search | **PARTIAL** | Brute force. `find_similar_segments` works. `find_neighbors` semantics are easy to misuse (bench bug). Uses `partial_cmp().unwrap()`, which panics on NaN. Not related to separation. |
| Cache-friendly memory access | **ABSENT** | Element-wise `[[ch, i]]` indexing loops throughout. The realtime path allocates on every callback. |
| Parallel processing (Rayon) | **PARTIAL** | Segments run in parallel. `BatchProcessor::with_threads` builds a new thread pool per call, and `batch_size` is dead code. |
| Progress bars | **PARTIAL** | The bar is created and then finished. It never advances. |
| Pre-trained model zoo `[x]` | **PLACEHOLDER** | 3 entries with `https://example.com/...` URLs and invented sizes. `download_model` always returns `NotSupported`. There are no models. `ModelRegistry` duplicates `ModelZoo`. |
| WebAssembly support `[x]` | **BROKEN** | Does not compile for wasm32 (3 errors). No backend could run in wasm anyway: `ort` with download-binaries does not target wasm32. |
| Candle backend implementation `[x]` | **PLACEHOLDER** + **BROKEN** | See above. `docs/IMPLEMENTATION.md:155` admits "requires candle-core updates". |
| Real-time CPAL integration `[x]` | **PLACEHOLDER** | See above. |
| Performance table (6x, 7x, 15x, "M1 MacBook Pro") | **UNSUBSTANTIATED** | No harness, no raw output, and nothing that could have been measured. Must be removed. |
| "Export from PyTorch Demucs" to ONNX | **ABSENT** | No exporter, no documented tensor contract, no STFT/iSTFT. HTDemucs and MDX-Net both need the spectrogram outside or around the graph. |
| Linfa / SmartCore / imbalanced-learn integration snippets | **ABSENT** | Those crates are not dependencies. The snippets import `charon::`, but the crate is `charon_audio`. `extract_audio_features`, `labels` and `x`/`y` are undefined. None of it compiles. |
| `cargo test --test audio_tests` | **ABSENT** | There is no `tests/` directory. |
| Docs link `docs.rs/charon` | **WRONG** | The crate is `charon-audio`. |
| crates.io badges | **UNVERIFIED** | Not checked whether 0.1.0 is published. If it is, the published crate has the same placeholders. |
| `realtime` example "conceptual" | **CONFIRMED** | Runs `Separator` on zeros, with `SeparatorConfig::default()` pointing at a nonexistent `model.onnx`, so it errors unless that file exists. |
| `examples/advanced.rs` | **PARTIAL** | Runs. It demonstrates the hint enum, iterator loops, and `thread::sleep` as "batch processing". Its "full pipeline" section is commented out. |

### 3.1 Manifest issues (`Cargo.toml`)

- No `Cargo.lock` committed, so builds are not reproducible. That is
  exactly what broke `main`.
- `ort` and `ort-sys` are unpinned RCs.
- `anyhow` and `env_logger` are normal dependencies but are used only by
  examples and docs; they belong in dev-dependencies. `num-traits`,
  `serde_json` and `candle-transformers` are never used.
- `cpal` is unconditional. It pulls in ALSA on Linux (a likely CI failure
  cause) and CoreAudio on macOS for every user, even when nothing runs
  in real time. It should be behind a `realtime` feature.
- `symphonia` with `features = ["all"]` enables every codec and container
  (including AAC/ISO-MP4). This should be a deliberate, documented choice
  (binary size, codec licensing), not a default.
- `ndarray` 0.15 is two major versions behind (0.17 is already in the
  graph via `ort`), so two ndarray copies get compiled.
- `web-sys`, `wasm-bindgen` etc. are declared for wasm32, but the wasm
  build is broken.
- `rust-version` is not declared.
- Only the `separate` and `realtime` examples are declared.
  `batch` and `advanced` are auto-discovered, which works but is
  inconsistent.
- `src/lib.rs:67` hard-codes the version string in a test.

### 3.2 Pipeline defects (independent of the placeholder)

1. **NaN at the first sample** (`src/processor.rs:227-247`). The fade-in
   starts at 0, so the overlap-add weight at sample 0 is 0, and 0/0
   gives NaN. The last segment's tail has the same problem where the
   fade-out reaches 0. Measured in section 2.3.
2. **Normalization is wrong for any real model**
   (`src/processor.rs:53-58, 83-86`). The input is z-scored with the
   global mean/std, and each source is de-normalized by adding the mix
   mean. Demucs does something similar internally, but it uses the mono
   reference std and does not add the mean to every stem. MDX-Net expects
   raw audio. This has to be per-model contract, not a global default.
3. **Shift ensemble uses circular roll** (`src/processor.rs:105-143`).
   The tail wraps onto the head, which creates a discontinuity. Demucs
   pads and uses random offsets within half a second.
4. **Linear fades, and weights normalized after the fact.** The window
   should be one that sums to a constant under the chosen overlap
   (Demucs uses a triangular weight with a power, applied per segment).
5. The whole song, all stems and all segment results are held in memory
   at once. A 10-minute song with 4 stems is about 800 MB of f32 before
   model activations.
6. `Stems` uses a `HashMap`, so stem order is nondeterministic (seen in
   section 2.3 output order).
7. The channel conversion `(n, m) if n > m` "downmix" keeps the first m
   channels; 5.1 to stereo drops the centre channel (vocals).
8. The realtime path locks mutexes and allocates in the audio callback.

## 4. What "working" has to mean

A separation library works when all of these hold:

1. A command with a real, downloadable, license-clean model produces
   stems from a real song.
2. Separation quality is measured: SDR per stem on a known reference set,
   compared with the same model run by its reference implementation on
   the same files. Target: within 0.1 dB of the reference. Any larger gap
   means a preprocessing bug.
3. `stems sum ~= mix` holds within a stated tolerance for models that
   are mixture-consistent, and it is tested.
4. It builds from a clean checkout on CI, with a committed lockfile, on
   Linux and macOS.

Numbers from vendors or papers (SDR, speedups) enter this plan only as
hypotheses. Each needs its own replication cell on this project's load
shape: specific songs, duration, sample rate, hardware, build hash and
feature set.

## 5. Minimum programme: v0.1.1 (1-2 days)

Goal: honest repository + one real working path. Scope is deliberately
narrow. Nothing here adds features.

### Model choice for the first working path

All facts below were read from primary sources on 2026-09-24. File sizes
come from GitHub/HF APIs; no model was downloaded yet. Every SDR figure
is "reported by X": a hypothesis, not a plan input.

| Candidate | I/O contract | Size | Weights license | Fit for v0.1.1 |
|---|---|---|---|---|
| **HTDemucs ONNX** (`huggingface.co/StemSplitio/htdemucs-onnx`, `htdemucs.onnx`) | `mix [1,2,343980]` -> `stems [1,4,2,343980]` (drums, bass, other, vocals). STFT is inside the graph, rewritten as Conv1d. Waveform in, waveform out. | 316 MB (165 MB fp16-weights variant) | Demucs code is MIT. The weights have no stated license; the "MIT" on the HF card is the re-hoster's interpretation. | **Recommended.** The existing waveform pipeline fits once `infer` is real. There is no STFT to get wrong. 4 stems match the current defaults. Risk: this is a third-party export, so parity against Python Demucs has to be checked on a clip. Whether normalization is inside the graph must be verified from the graph itself. |
| UVR MDX-Net `Kim_Vocal_2.onnx` (TRvlvr/model_repo release) | `input [B,4,3072,256]` spectrogram (L re/im, R re/im), n_fft 7680, hop 1024, periodic Hann, `center=True`, bins 0-2 zeroed, compensate 1.009. Params keyed by MD5 of the last 10,240,000 bytes. | 67 MB | **None.** The repo has no license. The UVR README asks for credit. | Second model in Phase A. Needs an exact `torch.stft` match, which is more risk than 2 days allows. Must not be redistributed; download from upstream only. |
| Open-Unmix umxhq / umx | Magnitude-spectrogram in/out, n_fft 4096, hop 1024 | no ONNX export published | **MIT** (Zenodo 3370489 / 3370486), the only clean weights found | Best license, but we would have to export it ourselves. Candidate for a redistributable default in the Phase B zoo. |
| BS-RoFormer / Mel-RoFormer ONNX (silverdaw on HF) | Spectrogram outside the graph, fixed 8 s window | 270 MB / 741 MB | Asserted MIT by the re-hoster; training data undocumented | Phase C quality tier, after the licensing is clarified. |

Decision for v0.1.1: HTDemucs ONNX (fp32), downloaded from upstream by
URL + SHA-256 at first use. It is not vendored, and the license caveat is
stated in the README.

Replication cells required before any quality or speed number appears
in the README:
- **Q1:** SDR of charon vs `demucs` (Python, same weights) on the same
  3 clips. Pass if the per-stem difference is at most 0.1 dB, or
  max |diff| is at most 1e-3 on the waveforms.
- **P1:** real-time factor and peak RSS for a 3-4 minute song on the M4
  Pro, CPU EP, with the build hash.

No quality number ships without Q1. No speed number ships without P1.

### Work items

| # | Item | Acceptance |
|---|---|---|
| M1 | Commit `Cargo.lock`; remove it from `.gitignore`. Pin `ort`/`ort-sys` to one exact RC, or move to the current RC and fix the two `?` sites (`map_err` into `CharonError::Model`). Prefer the current RC: rc.9 will not get fixes. | `cargo build` and `cargo test` pass on a clean clone. |
| M2 | Fix the cfg-gating: gate `test_config_builders` and the `onnx()` uses in examples on `ort-backend`, or give examples `required-features`. | `cargo test --no-default-features` and `cargo clippy --all-targets --all-features -D warnings` pass. |
| M3 | Manifest cleanup: move `anyhow`/`env_logger` to dev-dependencies; drop `num-traits`, `serde_json`, `candle-transformers`; move `cpal` + `realtime.rs` behind a `realtime` feature (off by default); declare `rust-version` after measuring it; version `0.1.1`; version test reads `CARGO_PKG_VERSION` against itself or is deleted. | `cargo tree` shows no unused crates; MSRV job passes. |
| M4 | Candle: either upgrade to a building version, or remove the feature and the placeholder code in 0.1.1 (it is not implemented anyway). Recommended: remove it now, reintroduce with a real model later. Same for `wasm.rs` (does not compile). Tombstone both in CHANGELOG. | `--all-features` builds. |
| M5 | Real ONNX inference for HTDemucs: build the `[1,2,343980]` tensor, call `session.run`, and map `[1,4,2,343980]` onto the stems in the model's order (drums, bass, other, vocals). Segment length is fixed at 343980 samples. Overlap-add follows the Demucs `apply_model` scheme. Apply normalization exactly as the reference does, with nothing added globally. Stem order comes from a per-model config, not `ModelConfig::default`. | `charon` example on a real song: vocals stem audibly isolated; replication cell Q1 passes. |
| M6 | Fix the overlap-add: a window with non-zero weight everywhere (or clamp the weight), padding instead of roll. | Unit test: output has no NaN/Inf on constant, silent and random input of lengths 1, seg-1, seg, seg+1, 3.5 seg. |
| M7 | Tests that mean something: (a) an identity-ONNX test (generated at test time) asserting output == input and no NaN, which exercises the tensor plumbing; (b) a WAV round-trip test; (c) an `#[ignore]` integration test that downloads the real model (checksum-verified), separates a short clip, and asserts SDR above a threshold. The threshold is set from a measured run, not guessed. | `cargo test` green; `cargo test -- --ignored` green locally with a network. |
| M8 | CI (GitHub Actions): fmt, clippy `-D warnings` over the feature matrix, tests on ubuntu + macos (install `libasound2-dev` only if `realtime` is enabled), docs build, MSRV job. The ignored model test runs on a weekly schedule with a cached model. | Green badge restored, pointing at a real workflow. |
| M9 | README rewrite: a **Status** section at the top (what works, what does not, what is planned); remove "state-of-the-art", "imbalanced learning", SIMD, KNN-as-feature, the performance table, the Linfa/SmartCore/imbalanced-learn snippets, and the `[x]` roadmap entries that are not true; fix the docs.rs link; document the one working command and the model license. Delete `docs/SUMMARY.md` and `docs/IMPLEMENTATION.md` (chat artifacts), or rewrite them. | Every sentence in the README is backed by code or a measurement in the repo. |
| M10 | CHANGELOG with 0.1.1 entry and tombstones for the removed placeholder APIs (Candle, wasm, model zoo URLs). Decide separately whether `performance.rs` (KNN, "SIMD", hints) stays. It is unrelated to separation. The recommendation is deprecation, to be decided by the owner in a separate pass. | Owner decision recorded. |

Estimate: 1.5-2 days, dominated by M5 (STFT contract correctness) and M7(c).

Out of scope for v0.1.1: Demucs, GPU, real time, wasm, model zoo, CLI binary.

## 6. Maximum programme: a finished product

The goal is "better, faster, more stable, cheaper than the competition".
The competition sets the bar. Each claim below becomes a measured row in
a published comparison table, or it does not go in the README.

### 6.1 Positioning

What a Rust implementation can credibly win on:
- **Distribution**: a single static binary / crate. Today the
  alternatives need a Python + PyTorch install (demucs,
  python-audio-separator, UVR).
- **Embedding**: a library API with no GIL, for DAW plugins, servers and
  mobile.
- **Memory and startup**: streaming chunked processing with bounded
  memory, and fast cold start.
- **Throughput per dollar on CPU / Apple Silicon**: ONNX Runtime with
  CoreML/CPU EPs, and batch servers without Python overhead.

What it will not win on by itself: separation quality. Quality comes from
the model weights. The quality claim is "bit-for-bit or within 0.1 dB of
the reference implementation for the same weights", never "better than
Demucs".

### 6.2 Phases

**Phase A: v0.2 "correct"** (1-2 weeks)
- A model contract trait: preprocess (STFT/none), tensor names/shapes,
  postprocess (iSTFT, masking, compensation), stem list. It is described
  in a manifest file shipped next to each model.
- MDX-Net family fully supported (several vocal/instrumental models).
- HTDemucs (4 and 6 stem) via ONNX: the STFT/iSTFT outside the graph
  must match `torch.stft` exactly. This is the hard part.
- Evaluation harness: `charon eval` computing SDR/SI-SDR (BSSEval v4
  style: 1 s frames, median) against references. Parity tests against
  the Python reference on a fixed set of clips, with a stored expected
  SDR per model and a tolerance.
- Streaming decode -> chunked inference -> streaming WAV/FLAC write,
  with bounded memory. Measure peak RSS.

**Phase B: v0.3 "usable"** (2-3 weeks)
- `charon` CLI binary: `charon separate <in> -o <dir> --model htdemucs`,
  `charon models list/download/verify`, `charon eval`. Progress that
  actually advances.
- Model zoo that exists: hosted manifests with SHA-256, license field,
  download over HTTPS, cache dir, offline mode. Only models whose weights
  license permits redistribution are listed; the rest get
  "download from upstream" instructions.
- Output formats: WAV 16/24/32f and FLAC. Plus stem naming, and
  mixture-consistency options (residual "other" = mix - sum).
- Execution providers: CPU, CoreML (macOS), CUDA (Linux). Selected at
  runtime, with fallback. Each is measured separately.

**Phase C: v0.4 "fast and cheap"** (2-4 weeks)
- Benchmark suite that measures what users pay for: real-time factor
  (audio seconds / wall seconds), peak RSS and cold start per model x EP
  x hardware, on a fixed song set. Stamp every report with build hash,
  feature set, EP, ORT version and hardware.
- Head-to-head against demucs (PyTorch CPU/MPS), python-audio-separator
  and demucs.cpp on the same machine and songs. Publish raw outputs.
- Optimisations, each measured one variable at a time: IO binding /
  preallocated tensors, batched segments, FP16/INT8 quantized variants
  (and their SDR cost), a thread-count policy, and an FFT backend
  (`realfft`) with SIMD.

**Phase D: v1.0 "product"**
- Stable API with semver guarantees. C ABI (`cdylib` + header) for
  DAW/plugin hosts. Python bindings (`pyo3`) as a drop-in for the
  Demucs CLI subset.
- Real-time / low-latency mode, only if a model supports it
  (small-context models). The latency and quality trade-off must be
  documented; HTDemucs is not real-time. The audio callback must be
  lock-free, with inference on a worker thread and ring buffers.
- WASM via `ort-web` or `tract`, only once a small model is measured to
  run acceptably in the browser.
- Fuzzing of the decoders/inputs, a regression corpus, and release
  automation (cargo-dist binaries for macOS/Linux/Windows).

### 6.3 What would cancel parts of this

- If a license check shows that the only good ONNX weights cannot be
  redistributed, the model zoo becomes "fetch from upstream", and the
  product value shifts to the runtime.
- If a measured head-to-head shows that the Rust + ORT path is not faster
  than PyTorch-MPS on Apple Silicon for HTDemucs, the "faster" claim
  is dropped. The product is then positioned on distribution and
  embedding only.
- If an existing maintained Rust crate already does real separation (see
  section 7), contributing to it may beat maintaining this one.

## 7. Competitor and model landscape

Last-activity dates and licenses are from GitHub, crates.io and PyPI,
read on 2026-09-24.

| Project | Status | License | Runtime | Notes |
|---|---|---|---|---|
| adefossez/demucs (maintained fork; the facebookresearch repo is archived) | active, PyPI 4.1.0 2026-07-11 | MIT | PyTorch | The quality reference. Weights now on HF as safetensors. |
| python-audio-separator | active, v0.47.0 2026-08-27 | MIT | torch + onnxruntime | The de-facto CLI for UVR/MDX/RoFormer models; CoreML for ONNX. |
| UVR GUI | last release 2023-09 | MIT per README (no LICENSE file) | Python | End-user GUI. |
| spleeter | stale (TF 2.12, Python < 3.12) | MIT | TensorFlow | Legacy. |
| sevagh/demucs.cpp | 2024-12 | MIT | C++17, Eigen | Native CPU Demucs. |
| **nikhilunni/demucs-rs** | active, v0.3.4 2026-03-10, 144 stars | Apache-2.0 | Burn + wgpu (Metal/Vulkan/WebGPU) | **Rust, real separation, and a VST3/CLAP plugin.** |
| **stem-splitter-core** | crates.io 1.2.0, 2026-04-13 | MIT/Apache | `ort` + HTDemucs (.ort), CoreML | **Rust, the same architecture this plan proposes for v0.1.1.** |
| IronHpc/UVR-rs (`uvr-*`) | v0.1.0 2026-09-14 | MIT | Burn / OpenVINO | Rust, VR models + BS-RoFormer. |

Consequence for the maximum programme, stated plainly: "the Rust
separation library" is not an open niche. At least two maintained
crates already do what v0.1.1 would do, and demucs-rs already ships a
GPU backend and a plugin. The "better/faster/cheaper than competitors"
goal therefore has to be earned on a specific axis, measured against
these crates, not against Python only:

- **Correctness as a product**: published parity tests (Q1-style) for
  every supported model. None of the competitors found publish
  reference-parity numbers.
- **Breadth of model contracts** under one API (HTDemucs + MDX +
  RoFormer + UMX) with model manifests and license metadata.
  python-audio-separator has the breadth but needs Python; the Rust
  crates each cover one family.
- **Bounded-memory streaming** and a measured RTF/RSS table against
  demucs-rs and stem-splitter-core on the same hardware.

If none of these differentiators survives a measured comparison in
Phase C, the recommended outcome is to stop, and to contribute the
evaluation harness upstream instead. That is a legitimate result, not a
failure.

Evaluation resources:
- museval v4 is 1 s frames, 512-tap distortion filters fitted once per
  track, frames aggregated with nanmedian and tracks with a median. There
  is no Rust port. The port is about 300 lines (Toeplitz autocorrelation
  plus a linear solve) and belongs in Phase A.
- MUSDB18 7-second preview set: 147 MB, AAC `.stem.mp4`, not HQ.
  MUSDB18(-HQ) is non-commercial / educational only. It is fine for CI
  evaluation but must not be redistributed in the repo.
- Reported SDRs (hypotheses only): demucs README reports htdemucs 9.00 /
  htdemucs_ft 9.20 dB (MUSDB-HQ test, mean). The python-audio-separator
  score table is not a test-set benchmark: its ~40 tracks include MUSDB
  train/validation tracks.

## 8. CV guidance

Until v0.1.1 is merged with a green CI badge and the end-to-end command
works from a clean clone, do not cite this project. After v0.1.1, the
defensible description is "Rust ONNX Runtime pipeline for HTDemucs
music source separation, with parity tests against the PyTorch reference
and CI". Every word of that is backed by a test. Given section 7, the
project stands out in a CV only once Phase A lands: multi-model
contracts plus a museval port with published parity.

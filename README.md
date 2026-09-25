# Charon Audio

[![Crates.io](https://img.shields.io/crates/v/charon-audio.svg)](https://crates.io/crates/charon-audio)
[![Documentation](https://docs.rs/charon-audio/badge.svg)](https://docs.rs/charon-audio)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Valkyra-Labs/charon-audio/workflows/CI/badge.svg)](https://github.com/Valkyra-Labs/charon-audio/actions)
[![Downloads](https://img.shields.io/crates/d/charon-audio.svg)](https://crates.io/crates/charon-audio)

**Rust music source separation pipeline for ONNX models.**

Charon runs the [HTDemucs](https://github.com/facebookresearch/demucs)
4-stem model (drums, bass, other, vocals) through ONNX Runtime, with
decoding, STFT/iSTFT, segmentation, overlap-add and output writing done
in Rust. No Python at run time. On macOS the network runs on the GPU
through CoreML. The pipeline reproduces Demucs 4.1.0 to float precision;
every number in this file comes from a recorded measurement, summarized
in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

## Status

| Area | State |
|---|---|
| HTDemucs 4-stem separation, CPU | **Works.** Parity with PyTorch Demucs: max sample difference 1.2e-4 on synthetic input, equal median SDR on MUSDB18 test previews ([measurements](docs/MEASUREMENTS.md)). |
| HTDemucs on the GPU (macOS, CoreML) | **Works** with the CoreML-target export (`coreml` feature): one CoreML partition, all nodes on the GPU. Output within 2e-6 of PyTorch. Separation 3x faster than the CPU path (5.6 s for a 193 s track); loading the compiled model costs 7 s per process. |
| Input: WAV, FLAC, MP3, OGG/Vorbis, AAC/M4A, ALAC, AIFF, CAF, MKV (Symphonia 0.6) | Works. Decoded samples are not clipped. Only WAV, FLAC and MP3 are exercised by tests or records. |
| Output: WAV 16/24-bit int, 32-bit float; FLAC 16/24-bit | Works, round-trip tested. |
| Resampling to the model rate (rubato) | Works, time-aligned (impulse tests). |
| Memory | Split export: 3.4 GB peak on CPU, 2.6 GB on CoreML (ONNX Runtime activation memory). In-graph export: 2.1 GB with its low-memory preset. |
| Speed, 193 s track on an Apple M4 Pro | Resident server on CoreML: 6.2-6.4 s per track (PyTorch on MPS, model resident: 7.3-7.6 s). Cold process: 14.1 s CoreML, 16.7 s CPU (PyTorch cold: 8.7 s MPS, 35.2 s CPU). Full table in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md). |
| Other models (MDX-Net, RoFormer, Open-Unmix, htdemucs_ft/6s) | **Not supported.** Each needs its own tensor contract and parity check. |
| CUDA, WebGPU | **Not supported.** CUDA is unmeasured (no hardware). WebGPU runs the in-graph export but slower than CPU. |
| Split export hosting | Not hosted yet. Produce the files with `tools/export/export_htdemucs.py` (PyTorch + demucs 4.1.0); hashes, verification and the hosting steps are in [docs/MODELS.md](docs/MODELS.md). |
| Real-time (CPAL) | Experimental, behind the `realtime` feature, not real-time safe, no tests. |
| Model download / zoo | Not implemented. `ModelZoo` and `models/manifest.json` hold names, hashes and URLs; fetch the files yourself (below). |
| `charon` binary | `charon separate`, and `charon serve`: a resident server that keeps the model (and the compiled CoreML model) loaded and answers jobs over a Unix socket. 6.3 s per 193 s track on CoreML with the server warm. |
| C ABI, Python bindings | Not in this release. |

The published crate `charon-audio 0.1.0` on crates.io is an earlier
version whose inference was a placeholder. Do not use it.

## Quick start

Three model files exist. The in-graph export runs on the CPU only and
can be downloaded. The split-transform exports (STFT/iSTFT in Charon)
are faster and currently have to be exported by you: one graph for the
CPU provider and one for CoreML (the CoreML provider needs the time
branch's convolutions tiled; that tiling costs 10% on the CPU).

### A. Download the in-graph export (CPU)

1. Get the model (316 MB, fp32) and check its hash:

```bash
curl -L -o htdemucs.onnx https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx
```

```bash
echo "68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74  htdemucs.onnx" | shasum -a 256 -c
```

2. Separate a file:

```bash
cargo run --release --example separate -- song.mp3 stems/ htdemucs.onnx
```

This writes `stems/{drums,bass,other,vocals}.wav` (32-bit float, 44.1 kHz
stereo). Options: `--format wav16|wav24|wav32|flac16|flac24`,
`--shifts N` (time-shift ensemble), `--max-speed` (ONNX Runtime defaults:
faster, 2.6x the memory).

### B. Export the split models and run them on the CPU or CoreML

```bash
python -m venv .venv && .venv/bin/pip install "demucs==4.1.0" torch onnx onnxruntime
```

```bash
.venv/bin/python tools/export/export_htdemucs.py htdemucs_split.onnx --target cpu
```

```bash
.venv/bin/python tools/export/export_htdemucs.py htdemucs_split_coreml.onnx --target coreml
```

The script checks each export against PyTorch before writing and prints
the SHA-256. With demucs 4.1.0 and torch 2.14.0 the CPU export hashes
to `6104c3de...8fe088f` every time; the CoreML export is structurally
identical between runs but its bytes differ (the measured artifact was
`782026bd...839607d`), see [docs/MODELS.md](docs/MODELS.md). Then:

```bash
cargo run --release --example separate -- song.mp3 stems/ htdemucs_split.onnx --split
```

```bash
cargo run --release --features coreml --example separate -- song.mp3 stems/ htdemucs_split_coreml.onnx --split
```

`--ep cpu|coreml|auto` picks the provider (default `auto`: CoreML if it
builds, else CPU). The first CoreML run compiles the model (about 30 s)
into `coreml-cache/<model hash>/` next to the model file; later
processes load it in about 7 s.

### C. The `charon` binary and the resident server

```bash
cargo build --release --features coreml --bin Charon
```

One-shot (same as the example):

```bash
target/release/charon separate song.mp3 -o stems/ --model htdemucs_split_coreml.onnx --format flac24
```

Resident: the server pays the model load once; each job then costs only
decode + separation + write (6.3 s for a 193 s track on an M4 Pro):

```bash
target/release/charon serve --model htdemucs_split_coreml.onnx
```

```bash
target/release/charon separate song.mp3 -o stems/
```

`charon separate` uses the server when its socket answers (default
`$TMPDIR/charon-$USER.sock`, or `--socket`), otherwise it runs
in-process with `--model`. `charon ping` and `charon stop` talk to the
server. The protocol is one JSON object per line over a Unix socket.

### Library

```rust
use charon_audio::{BitDepth, Separator, SeparatorConfig, StemFormat};

fn main() -> anyhow::Result<()> {
    let separator = Separator::new(SeparatorConfig::htdemucs("htdemucs.onnx"))?;
    let stems = separator.separate_file("song.mp3")?;
    stems.save_all_as("stems", StemFormat::Flac(BitDepth::Int24))?;
    Ok(())
}
```

`SeparatorConfig::htdemucs` sets the in-graph model contract (input
`mix` `[1, 2, 343980]`, output `stems` `[1, 4, 2, 343980]`, stem order
drums, bass, other, vocals) and the low-memory ONNX Runtime preset.
`SeparatorConfig::htdemucs_split` sets the split contract (`mix` and a
complex-as-channels spectrogram `spec` `[1, 4, 2048, 336]` in; `time`
`[1, 4, 2, 343980]` and `spec_out` `[1, 4, 4, 2048, 336]` out; Charon
does `HTDemucs._spec`/`_ispec` and sums the branches). The processing
settings follow Demucs: 7.8 s segments, 25% overlap, triangular
overlap-add weights, normalization by the mono reference mean and
standard deviation.

```rust
use charon_audio::{OnnxOptions, SeparatorConfig};

let mut config = SeparatorConfig::htdemucs("htdemucs.onnx").with_shifts(2);
config.model.onnx = OnnxOptions::max_speed();
```

## Model license

The HTDemucs code is MIT. The weights carry no license statement from
their authors; the ONNX export on Hugging Face is re-hosted by a third
party under MIT on their own interpretation. Charon does not vendor or
redistribute the weights. Check the terms for your use before shipping
them.

## Measurements

Every record carries build hash, model hash, versions, host and method;
[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) has the tables and
`tools/parity/` the scripts. Summary, Apple M4 Pro:

- Parity with PyTorch demucs 4.1.0 (`shifts=0`): max sample difference
  1.2e-4 on synthetic clips, 9.3e-4 on three full-length tracks;
  identical median SDR to two decimals on 50 MUSDB18 test previews.
- MUSDB18 7-second test previews, whole-signal SDR (not museval; not
  comparable with published tables): drums 9.50, bass 9.04, other 5.19,
  vocals 8.88 dB. Mixture-as-estimate baseline: -4.1 to -6.9 dB.
- Split exports, 193 s track: separation 16.6 s on CPU, 5.6 s on CoreML
  (one partition, RTF 34); end to end 17.1 s CPU, 13.4 s CoreML; peak
  RSS 3.4 GB CPU, 2.6 GB CoreML.
- In-graph export, 193 s track: 2.12 GB peak (low-memory preset),
  5.58 GB (`max_speed`); 26.4 s end to end.
- Head-to-head on the same track and machine against PyTorch (CPU,
  MPS, cold and resident), stem-splitter-core and demucs-rs: resident
  Charon on CoreML 6.2-6.4 s, PyTorch MPS resident 7.3-7.6 s, PyTorch
  MPS cold 8.7 s, demucs-rs 13.7 s (with NaN output), Charon cold
  14.1 s CoreML / 16.7 s CPU, stem-splitter-core 27.3 s, PyTorch CPU
  35.2 s. Quality on MUSDB previews: Charon equals PyTorch,
  stem-splitter-core 0.2-0.3 dB lower, demucs-rs lower with 6 of 50
  tracks NaN.

## Building

Rust 1.89 or later (`rust-version` in `Cargo.toml`, checked in CI).
`ort` downloads a prebuilt ONNX Runtime 1.28 at build time; no system
ONNX Runtime is needed.

```bash
cargo test
```

The real-model regression test needs the model:

```bash
CHARON_HTDEMUCS_MODEL=/path/to/htdemucs.onnx cargo test --release -- --ignored
```

Features: `ort-backend` (default), `coreml` (CoreML execution provider,
macOS), `realtime` (CPAL input, experimental), `ep-experimental`
(CoreML + WebGPU for `examples/ep_probe`).

## Repository layout

- `src/audio.rs`: decoding (Symphonia), resampling (rubato), WAV/FLAC
  writing.
- `src/models.rs`: ONNX Runtime session, model contracts, session
  options, execution providers.
- `src/stft.rs`: Demucs STFT/iSTFT (`_spec`/`_ispec`) with realfft.
- `src/processor.rs`: segmentation, overlap-add, shifts, normalization.
- `src/separator.rs`: `Separator`, `SeparatorConfig`, `Stems`.
- `src/bin/charon.rs`: the CLI and the resident server.
- `tests/`: identity-model pipeline tests, real-model regression tests.
- `tools/parity/`: the measurement scripts; `tools/export/`: the
  split-transform ONNX export.
- `models/`: manifest and checksums of the supported model files.
- `docs/`: [MEASUREMENTS](docs/MEASUREMENTS.md), [MODELS](docs/MODELS.md),
  [IMPLEMENTATION](docs/IMPLEMENTATION.md), [CONTRIBUTING](docs/CONTRIBUTING.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE). Model weights are not covered, see above.

## Acknowledgements

- [Demucs](https://github.com/facebookresearch/demucs) (Rouard, Massa,
  Défossez): the model and the reference pipeline.
- [StemSplitio/htdemucs-onnx](https://huggingface.co/StemSplitio/htdemucs-onnx):
  the ONNX export.
- [ONNX Runtime](https://onnxruntime.ai/) and the [`ort`](https://github.com/pykeio/ort) crate.
- [Symphonia](https://github.com/pdeljanov/Symphonia), [rubato](https://github.com/HEnquist/rubato),
  [hound](https://github.com/ruuda/hound), [flacenc](https://github.com/yotarok/flacenc-rs).

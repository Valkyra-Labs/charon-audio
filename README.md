# charon-audio

Rust music source separation pipeline for ONNX models.

charon runs the [HTDemucs](https://github.com/facebookresearch/demucs)
4-stem model (drums, bass, other, vocals) through ONNX Runtime, with
decoding, segmentation, overlap-add and output writing done in Rust.
No Python at run time. The separation pipeline reproduces Demucs 4.1.0
to float precision; every number in this file has a measurement record
under [`docs/parity/`](docs/parity/).

## Status

| Area | State |
|---|---|
| HTDemucs 4-stem separation (ONNX, CPU) | **Works.** Parity with PyTorch Demucs: max sample difference 1.2e-4 on synthetic input, equal median SDR on MUSDB18 test previews. See [Q1](docs/parity/2026-09-24-htdemucs-q1.md), [MUSDB7](docs/parity/2026-09-24-htdemucs-musdb7.md), [real tracks](docs/parity/2026-09-24-htdemucs-real-tracks.md). |
| Input: WAV, FLAC, MP3, OGG/Vorbis, AAC/M4A, ALAC, AIFF, CAF, MKV (Symphonia 0.6) | Works. Decoded samples are not clipped. Only WAV, FLAC and MP3 are exercised by tests or records. |
| Output: WAV 16/24-bit int, 32-bit float; FLAC 16/24-bit | Works, round-trip tested. |
| Resampling to the model rate (rubato) | Works, time-aligned (impulse tests). |
| Memory | 2.1 GB peak with the default preset, 5.6 GB with `max_speed()`. [Record](docs/parity/2026-09-24-memory.md). |
| Speed | About 7x real time on an Apple M4 Pro CPU (193 s track in 26.9 s including decode and write). [Record](docs/parity/2026-09-24-head-to-head.md). |
| Other models (MDX-Net, RoFormer, Open-Unmix, htdemucs_ft/6s) | **Not supported.** Each needs its own tensor contract and parity check. |
| GPU / CoreML / CUDA | **Not supported.** The ORT 1.28 CoreML provider fails on this export in every configuration tried (record in the memory document). CUDA is unmeasured. |
| Real-time (CPAL) | Experimental, behind the `realtime` feature, not real-time safe, no tests. |
| Model download / zoo | Not implemented. `ModelZoo` holds metadata only; download the model yourself (below). |
| CLI binary, C ABI, Python bindings | Not in this release. |

The published crate `charon-audio 0.1.0` on crates.io is an earlier
version whose inference was a placeholder. Do not use it.

## Quick start

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

`SeparatorConfig::htdemucs` sets the model contract (input `mix`
`[1, 2, 343980]`, output `stems` `[1, 4, 2, 343980]`, stem order drums,
bass, other, vocals) and the low-memory ONNX Runtime preset. The
processing settings follow Demucs: 7.8 s segments, 25% overlap,
triangular overlap-add weights, normalization by the mono reference mean
and standard deviation.

```rust
use charon_audio::{OnnxOptions, SeparatorConfig};

let mut config = SeparatorConfig::htdemucs("htdemucs.onnx").with_shifts(2);
config.model.onnx = OnnxOptions::max_speed();
```

## Model license

The HTDemucs code is MIT. The weights carry no license statement from
their authors; the ONNX export on Hugging Face is re-hosted by a third
party under MIT on their own interpretation. charon does not vendor or
redistribute the weights. Check the terms for your use before shipping
them.

## Measurements

All records are in [`docs/parity/`](docs/parity/) with build hash,
model hash, versions, host and method. Summary, Apple M4 Pro, CPU:

- Parity with PyTorch demucs 4.1.0 (`shifts=0`): max sample difference
  1.2e-4 on synthetic clips, 9.3e-4 on three full-length tracks;
  identical median SDR to two decimals on 50 MUSDB18 test previews.
- MUSDB18 7-second test previews, whole-signal SDR (not museval; not
  comparable with published tables): drums 9.50, bass 9.04, other 5.19,
  vocals 8.88 dB. Mixture-as-estimate baseline: -4.1 to -6.9 dB.
- Memory, 193 s track: 2.12 GB peak (default preset), 5.58 GB
  (`max_speed`). Throughput cost of the default preset: 11%.
- Head-to-head on the same track and machine, see the
  [head-to-head record](docs/parity/2026-09-24-head-to-head.md).

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

Features: `ort-backend` (default), `realtime` (CPAL input, experimental).

## Repository layout

- `src/audio.rs`: decoding (Symphonia), resampling (rubato), WAV/FLAC
  writing.
- `src/models.rs`: ONNX Runtime session, model contract, session options.
- `src/processor.rs`: segmentation, overlap-add, shifts, normalization.
- `src/separator.rs`: `Separator`, `SeparatorConfig`, `Stems`.
- `tests/`: identity-model pipeline tests, real-model regression test.
- `tools/parity/`: scripts that produced the records in `docs/parity/`.
- `docs/audit/`: the audit and plan this release was built against.

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

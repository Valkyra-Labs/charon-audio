# Head-to-head: charon vs PyTorch Demucs, stem-splitter-core, demucs-rs, 2026-09-24

One machine, one track, one track set, every tool run alone. Speed and
memory are whole-process CLI runs (decode + separate + write). Quality is
the same whole-signal SDR as the MUSDB7 record, on the same 50 MUSDB18
test previews.

## Stamp

- Host: Apple M4 Pro (14 cores, GPU), 24 GB, macOS 26.6. Nothing else
  running during timing rows.
- charon: commit `aefd582`, `cargo build --release --example separate`,
  default features. ort `=2.0.0-rc.13`, ONNX Runtime 1.28.0, CPU EP.
  Model `htdemucs.onnx` (SHA-256 `68d0bf16...5fcc5e74`).
- stem-splitter-core 1.2.0 (crates.io, `cargo install`), its own
  `htdemucs_ort_v1` model (gentij/htdemucs-ort `.ort`), ort 2.0.0-rc.11.
  CPU EP (CoreML is listed as auto-detected, the log did not show it).
- demucs-rs commit `5d9f61a` (2026-08-01), `cargo build --release -p
  demucs-cli`, Burn 0.20.1 on wgpu/Metal, model `htdemucs` (its own
  84 MB weights download).
- PyTorch: demucs 4.1.0, torch 2.14.0, `python -m demucs -n htdemucs
  --shifts 0`, devices `cpu` and `mps`, default output (int16,
  clip-mode rescale).
- Track: 192.6 s stereo 44.1 kHz float WAV (`audioknap-blockbuster`).
  Method: `tools/parity/compare_tools.sh`, `/usr/bin/time -l`. Two
  passes, both shown. First runs that download models were discarded.

## Speed and memory, 193 s track

| tool | device | wall s (pass 1 / 2) | peak RSS MB (pass 1 / 2) | output finite |
|---|---|---|---|---|
| charon, default (low memory) | CPU | 26.9 / 26.4 | 2106 / 2099 | yes |
| charon, `--max-speed` | CPU | 22.0 / 22.0 | 5573 / 5549 | yes |
| stem-splitter-core 1.2.0 | CPU | 27.7 / 27.9 | 4936 / 4795 | yes |
| PyTorch demucs 4.1.0 | CPU | 35.5 / 35.6 | 2339 / 2595 | yes |
| demucs-rs 5d9f61a | Metal (wgpu) | 15.1 / 15.2 | 872 / 876 | **no**: 2,579,850 / 343,980 NaN frames of 8,494,848 |
| PyTorch demucs 4.1.0 | MPS | 9.0 / 9.1 | 1958 / 1961 | yes |

Reading, CPU only: charon is the fastest CPU tool on this track (22.0 s
at ORT defaults; 26.4 s with the low-memory preset) and, with the
preset, uses 2.1 GB against 4.8-4.9 GB for stem-splitter-core and
2.3-2.6 GB for PyTorch CPU. GPU: PyTorch on MPS is 2.4x faster than
charon's best and uses less memory than charon's fast preset. demucs-rs
on Metal is fast and small, but its output on this track contained NaN
in 30% (pass 1) and 4% (pass 2) of frames; the count differs between
runs, so it is not deterministic. Its 5 s warm-up output was finite.

No GPU path exists in charon (CoreML failed, see the memory record), so
"faster than PyTorch" holds for CPU only and must be stated that way.

## Agreement with the PyTorch reference, same track

Reference: `demucs.api.Separator("htdemucs", shifts=0)` output
(`tools/parity/reference.py torch`), float, unclipped. Agreement SDR
`10*log10(sum(ref^2)/sum((ref-est)^2))`; pipeline agreement, not quality.

| tool | drums | bass | other | vocals |
|---|---|---|---|---|
| charon (both presets, identical output) | 77.1 dB, max 6.0e-4 | 66.5 dB, 9.3e-4 | 80.5 dB, 4.7e-4 | 77.8 dB, 1.3e-4 |
| PyTorch CLI, `--float32 --clip-mode none`, CPU | 76.7 dB, max 6.1e-2 | exact | 83.4 dB, 2.9e-2 | exact |
| PyTorch CLI, default output (int16, `clip-mode rescale`) | 23.5 dB | 69.3 dB | 28.3 dB | 66.1 dB |
| stem-splitter-core | 15.8 dB | 9.6 dB | 18.2 dB | 10.3 dB |
| demucs-rs | NaN in output | | | |

Notes: the demucs CLI's own float output differs from the API by up to
6e-2 at single samples in drums/other (at 111.4 s and 134.6 s); charon
matches the API more closely than the CLI does. The CLI's default
int16 + rescale output rescales the loud stems, which is why it lands at
23-28 dB. stem-splitter-core runs a different export and pipeline; its
10-18 dB agreement means it produces a different separation, which the
MUSDB numbers below quantify.

## Quality, MUSDB18 test previews (50 tracks, median whole-signal SDR, dB)

Method: `tools/parity/musdb_eval.py` (charon, PyTorch) and
`tools/parity/musdb_eval_tool.py` (others). Non-finite samples in a
tool's output are counted and zeroed before scoring.

| tool | drums | bass | other | vocals | tracks with non-finite output |
|---|---|---|---|---|---|
| charon (shifts 1, default) | **9.50** | **9.04** | **5.19** | **8.88** | 0 |
| PyTorch demucs 4.1.0, shifts 0 | 9.50 | 9.04 | 5.19 | 8.88 | 0 |
| stem-splitter-core 1.2.0 | 9.31 | 8.76 | 4.89 | 8.55 | 0 |
| demucs-rs 5d9f61a (Metal) | 9.33 | 8.94 | 5.13 | 7.48 | 6 of 50 (entire track NaN) |
| mixture as estimate | -4.13 | -6.89 | -4.98 | -4.49 | |

charon equals the PyTorch reference and is 0.19-0.33 dB above
stem-splitter-core on every stem. demucs-rs is 0.06-0.17 dB below on
drums/bass/other and 1.40 dB below on vocals, with 6 tracks lost to NaN.
The metric is not museval and the clips are 7 s AAC excerpts; see the
MUSDB7 record for the caveats.

## What this does and does not support

- Supported claim: on CPU, charon is faster than PyTorch Demucs and
  stem-splitter-core on this track, uses less memory than both with its
  default preset, and separates better than stem-splitter-core on MUSDB
  previews (same weights family, different export). Its output equals
  the PyTorch reference.
- Not supported: any claim against GPU paths. PyTorch on MPS is faster.
- One track and one machine for the timing rows. Two passes agree within
  3%, but this is not a benchmark suite.

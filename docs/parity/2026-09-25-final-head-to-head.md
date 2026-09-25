# Final head-to-head, 2026-09-25: charon 0.1.1 vs PyTorch Demucs, stem-splitter-core, demucs-rs

One machine, one 193 s track, every tool alone, two passes; quality on
the same 50 MUSDB18 test previews for every tool. This supersedes the
2026-09-24 head-to-head record for charon's rows (new exports, CoreML,
resident server); the competitor rows are re-measured here.

## Stamp

- Host: Apple M4 Pro (14 CPU cores, GPU), 24 GB, macOS 26.6, nothing
  else running during timing rows.
- charon commit `e6896c4` + the `charon` binary commit (this record's
  build): `cargo build --release --features coreml --bin charon`. ort
  `=2.0.0-rc.13`, ONNX Runtime 1.28.0. Models: `htdemucs_split.onnx`
  (CPU, SHA-256 `6104c3de...`), `htdemucs_split_coreml.onnx` (CoreML,
  SHA-256 `782026bd...`), both from `tools/export/export_htdemucs.py`
  with demucs 4.1.0 / torch 2.14.0.
- stem-splitter-core 1.2.0 (`cargo install`), its `htdemucs_ort_v1`
  model, ort rc.11. demucs-rs commit `5d9f61a`, Burn 0.20.1 on Metal.
  PyTorch: demucs 4.1.0, torch 2.14.0, `python -m demucs -n htdemucs
  --shifts 0` (cold rows), `demucs.api.Separator` in one process
  (resident rows).
- Track: 192.6 s stereo 44.1 kHz float WAV. Script:
  `tools/parity/compare_tools.sh`; `/usr/bin/time -l` for cold rows;
  client wall time for `charon serve` jobs (the server does decode,
  separation and writing); in-process timing for PyTorch resident.
- Pass 1 contained two first-time compiles (charon CoreML 32.9 s,
  stem-splitter 45.1 s: both compile a CoreML model on first use in a
  fresh cache) and is reported for completeness; pass 2 is the clean
  pass.

## Speed and memory, 193 s track

| tool | device | mode | wall, pass 2 (pass 1) | peak RSS | output |
|---|---|---|---|---|---|
| **charon, `serve`** | CoreML | **resident** | **6.15 / 6.27 / 6.39 s** (6.30 / 6.37 / 6.44) | server 1.5-1.6 GB steady, client 8 MB | finite, = PyTorch to 2e-5 |
| PyTorch demucs 4.1.0 | MPS | resident (`demucs.api`) | 7.32 / 7.35 / 7.59 s (7.40 / 7.39 / 7.66) | not measured | reference |
| PyTorch demucs 4.1.0 | MPS | cold CLI | 8.67 s (9.08) | 1.96 GB | reference |
| demucs-rs 5d9f61a | Metal | cold CLI | 13.70 s (13.99) | 0.88 GB | **NaN in 4,471,740 of 8,494,848 frames** (pass 1; earlier record: 30% and 4%) |
| **charon, one-shot** | CoreML | cold CLI | **14.08 s** (32.86: first compile) | 2.6 GB | finite, = PyTorch to 2e-5 |
| **charon, one-shot** | CPU | cold CLI | **16.70 s** (17.83) | 3.5 GB | finite, = PyTorch to 9e-4 |
| stem-splitter-core 1.2.0 | CPU (CoreML attempt, see note) | cold CLI | 27.28 s (45.08: first compile) | 4.9 GB | finite, 10-18 dB from PyTorch |
| PyTorch demucs 4.1.0 | CPU | resident | 33.79 / 33.90 s | not measured | reference |
| PyTorch demucs 4.1.0 | CPU | cold CLI | 35.20 s (35.28) | 2.6 GB | reference |

Note on stem-splitter-core: its first run compiled a CoreML model
(45 s) but its steady-state time and 4.9 GB RSS match a CPU run of the
same architecture; whether its CoreML partition executes anything is
not visible from outside and was not investigated further.

## Agreement with PyTorch on this track

Reference: `python -m demucs --float32 --clip-mode none` output (pass 1
files). Agreement SDR `10*log10(sum(ref^2)/sum((ref-est)^2))`, max
sample difference; pipeline agreement, not separation quality. The
reference itself deviates from `demucs.api` by up to 6e-2 on drums and
other at two isolated samples (see the 2026-09-24 head-to-head record),
which caps the drums/other rows for every tool.

| tool | drums | bass | other | vocals |
|---|---|---|---|---|
| charon CoreML (one-shot and resident, identical) | 76.7 dB, 6.1e-2 | 113.1 dB, 2.0e-5 | 83.4 dB, 2.9e-2 | 116.1 dB, 4.0e-6 |
| PyTorch MPS resident | 76.7 dB, 6.1e-2 | 114.1 dB, 3.6e-6 | 83.4 dB, 2.9e-2 | 117.0 dB, 8.4e-7 |
| charon CPU | 73.9 dB, 6.1e-2 | 66.5 dB, 9.2e-4 | 78.7 dB, 2.9e-2 | 77.8 dB, 1.3e-4 |
| stem-splitter-core | 15.8 dB | 9.6 dB | 18.2 dB | 10.3 dB |
| demucs-rs | NaN | NaN | NaN | NaN |

charon on CoreML agrees with PyTorch as closely as PyTorch on MPS agrees
with PyTorch on CPU.

## Quality, MUSDB18 test previews (50 tracks, median whole-signal SDR, dB)

From the MUSDB records (`musdb_eval.py`, `musdb_eval_tool.py`); charon's
three paths were measured separately and are identical to two decimals.

| tool | drums | bass | other | vocals | tracks with non-finite output |
|---|---|---|---|---|---|
| charon (CPU, CoreML, in-graph) | 9.50 | 9.04 | 5.19 | 8.88 | 0 |
| PyTorch demucs 4.1.0 | 9.50 | 9.04 | 5.19 | 8.88 | 0 |
| stem-splitter-core 1.2.0 | 9.31 | 8.76 | 4.89 | 8.55 | 0 |
| demucs-rs 5d9f61a | 9.33 | 8.94 | 5.13 | 7.48 | 6 of 50 |
| mixture as estimate | -4.13 | -6.89 | -4.98 | -4.49 | |

Not museval; 7 s AAC excerpts; see the MUSDB7 record for caveats.

## Reading

- With the model resident, charon is the fastest tool measured on this
  machine: 6.2-6.4 s per 193 s track (RTF 30), against 7.3-7.6 s for
  PyTorch on MPS with the same warm-model treatment, at equal
  separation quality and with output that matches PyTorch to 2e-5.
- Cold, charon on CoreML (14.1 s) trails PyTorch MPS (8.7 s) because
  macOS compiles the CoreML program on load: 7 s when Apple's own
  compiled-model cache is warm, about 27 s when it is not (the ORT
  on-disk cache does not avoid this; the recompile happens even when
  ORT's cache entry is untouched). This is outside ORT and charon; the
  server exists to pay it once.
- On CPU, charon (16.7 s) is 2.1x faster than PyTorch (35.2 s) and 1.6x
  faster than stem-splitter-core (27.3 s), at equal quality to PyTorch
  and 0.2-0.3 dB better than stem-splitter-core.
- demucs-rs is fast and small but its Metal path produced NaN in more
  than half the frames of this track in this pass, and in 6 of 50 MUSDB
  previews; its numbers are not usable as they stand.
- Memory: charon's server holds 1.5-1.6 GB steady after jobs; the cold
  CPU path peaks at 3.5 GB (ONNX Runtime activation memory, the last
  open item on the CPU side).

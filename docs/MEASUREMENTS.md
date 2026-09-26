# Measurements behind the README

Every number in the README comes from a run whose build hash, model
hash, tool versions, host and method were recorded. The full records
(about 1,500 lines with per-cell tables) are kept outside the repository;
this file is the public summary with the same numbers, and the scripts
that produced them are in `tools/parity/`.

Host for everything below: Apple M4 Pro (12 CPU cores: 8 performance,
4 efficiency; GPU), 24 GB,
macOS 26.6. Charon: ort `=2.0.0-rc.13` (ONNX Runtime 1.28.0), release
build, default features plus `coreml` where stated. Reference: demucs
4.1.0, torch 2.14.0. Dates: 2026-09-24 and 2026-09-25.

## Parity with PyTorch Demucs

Method: the same input to Charon and to `demucs.api.Separator("htdemucs",
shifts=0, overlap=0.25, split=True)`; max absolute sample difference and
agreement SDR `10*log10(sum(ref^2)/sum((ref-est)^2))` per stem. Agreement
SDR measures how closely two pipelines agree; it is not separation
quality.

| input | Charon path | max diff (worst stem) | agreement |
|---|---|---|---|
| 3 synthetic clips (5, 20, 31.3 s) | in-graph export, CPU | 1.2e-4 | 66.8-90.0 dB |
| 3 synthetic clips | split export, CPU | 1.2e-4 | 66.7-90.0 dB |
| 3 synthetic clips | split export, CoreML | 1.8e-6 | 81.6-128 dB |
| 193 s real track | split export, CPU | 9.2e-4 | 66.5-80.5 dB |
| 193 s real track | split export, CoreML | 1.3e-5 | 114-121 dB |
| 3 real tracks, 110-193 s | in-graph export, CPU | 9.3e-4 | 66.0-96.4 dB |

Acceptance criterion: max difference at most 1e-3. STFT/iSTFT
(`src/stft.rs`) are checked separately against `torch.stft` fixtures at
1e-5 relative (`tests/fixtures/make_stft_fixture.py`). The two ignored
tests in `tests/htdemucs.rs` re-check a 9 s fixture against PyTorch
envelopes on every run with a model present (worst frame-RMS difference
1.9e-5 on CPU, 2.7e-7 on CoreML).

## Separation quality, MUSDB18 test previews

50 tracks of the MUSDB18 7-second preview set, test split only
(HTDemucs trained on the train split). Metric: whole-signal SDR per
stem, median over tracks. Not museval BSSEval v4 and 7 s AAC excerpts,
so not comparable with published MUSDB tables. Script:
`tools/parity/musdb_eval.py`, `tools/parity/musdb_eval_tool.py`.

| tool | drums | bass | other | vocals | non-finite output |
|---|---|---|---|---|---|
| Charon (in-graph CPU, split CPU, split CoreML: identical) | 9.50 | 9.04 | 5.19 | 8.88 | 0 tracks |
| PyTorch demucs 4.1.0 | 9.50 | 9.04 | 5.19 | 8.88 | 0 |
| stem-splitter-core 1.2.0 | 9.31 | 8.76 | 4.89 | 8.55 | 0 |
| demucs-rs 5d9f61a (Metal) | 9.33 | 8.94 | 5.13 | 7.48 | 6 of 50 |
| mixture as estimate | -4.13 | -6.89 | -4.98 | -4.49 | |

Time-shift ensemble (`--shifts`): 2 shifts +0.04 dB mean, 4 shifts
+0.06 dB (bass +0.18, vocals none), for 2x and 4x the time. Default
stays off.

## Speed and memory, 193 s track, whole runs (decode + separate + write)

Cold rows: one process each, `/usr/bin/time -l`, two passes (second,
clean pass shown; first-time CoreML compiles excluded). Resident rows:
model loaded once, three consecutive jobs. Script:
`tools/parity/compare_tools.sh`.

| tool | device | mode | wall | peak RSS |
|---|---|---|---|---|
| Charon `serve` | CoreML | resident | 6.15 / 6.27 / 6.39 s | server 1.5-1.6 GB, client 8 MB |
| PyTorch demucs 4.1.0 | MPS | resident (`demucs.api`) | 7.32 / 7.35 / 7.59 s | |
| PyTorch demucs 4.1.0 | MPS | cold | 8.67 s | 1.96 GB |
| demucs-rs 5d9f61a | Metal | cold | 13.70 s | 0.88 GB (NaN in 53% of frames) |
| Charon one-shot | CoreML | cold | 14.08 s | 2.6 GB |
| Charon one-shot | CPU | cold | 16.70 s | 3.5 GB |
| stem-splitter-core 1.2.0 | CPU | cold | 27.28 s | 4.9 GB |
| PyTorch demucs 4.1.0 | CPU | resident | 33.8 s | |
| PyTorch demucs 4.1.0 | CPU | cold | 35.20 s | 2.6 GB |
| Charon 0.1.1 before this work (in-graph export, CPU) | CPU | cold | 26.4 s | 2.1 GB |

Separation only (`examples/profile`), split exports: 16.6 s on CPU
(RTF 11.6), 5.6 s on CoreML (RTF 34). A cold CoreML process spends
7 s loading the compiled model when macOS's compiled-model cache is
warm and about 27 s when it is not; this happens outside ONNX Runtime
(its on-disk cache entry is untouched either way), and the resident
server exists to pay it once.

## What the numbers do not support

- No claim against GPU tools in cold mode: PyTorch on MPS starts faster.
- No claim about CUDA or Linux: unmeasured.
- One track and one machine for the timing rows; passes agree within
  3%, but this is not a benchmark suite.

## Memory and provider experiments (summary)

- In-graph export: ONNX Runtime's constant folding materializes 3.8 GB of
  STFT index tensors at load; `OnnxOptions::low_memory()` (folding and
  memory pattern off) takes peak RSS from 5.58 GB to 2.12 GB for 11%
  throughput. fp16 weights change nothing.
- CoreML fails on the in-graph export in four configurations; it runs
  the split export, and runs it as a single partition only when the
  time branch's convolutions are 4-D with spatial axes of at most 16384
  samples (the `--target coreml` export tiles them). Partition count
  26 -> 1 took the per-segment time from 0.235 s to 0.157 s.
- Tried and dropped, each measured: WebGPU (runs, slower than CPU),
  Neural Engine (slower than the GPU), NeuralNetwork model format
  (190 s per run), batch-2 export (slower on both providers), CoreML
  FastPrediction specialization (no change), fp16 conversion (invalid
  graph).

## TIGER-DnR music branch (0.1.2, 2026-09-25)

Build: branch `release/0.1.2` on top of `6831e48` (uncommitted working
tree at the time), release, features `ort-backend,decode,aac,coreml`.
Model `tiger_music.onnx` SHA-256 `bb52541d...`, from checkpoint
`dd1c696e...`. Reference: the TIGER repository at `9f18d4a`, torch
2.14.0, ONNX Runtime 1.30.0 in Python for the export check. Host as
above (M4 Pro, macOS 26.6).

Parity:

| check | result |
|---|---|
| ONNX export vs PyTorch module, one 12 s window, real audio (DnR test mixture 466; a cinematic music track) | 105.8 dB / 95.9 dB agreement, max waveform difference 1.2e-6 / 2.0e-5 |
| Full Charon pipeline (Rust STFT/iSTFT, windowing) vs PyTorch on `synth_9s.flac` (`tests/htdemucs.rs`) | max 50 ms frame-RMS difference 1.7e-5 |
| Charon with the authors' window layout (12 s windows, 4 s hop, uniform blend) vs the authors' `wav_chunk_inference` run locally on CPU, 60 s mixture | 84.1 dB agreement away from the first and last 8 s (the authors pad the file edges with zeros; Charon does not), max difference 2.2e-5 |
| The same with the stride computed in f32 and truncated (the bug fixed in 0.1.2) | 25.5 dB: windows drift by a sample each, and the model is sensitive to window placement |

Speed and memory, 60 s mono mixture, CPU:

| setting | wall | peak RSS |
|---|---|---|
| 12 s windows, 0.5 overlap, memory pattern on | 41.0 s | 6.5 GB |
| same, memory pattern off (the preset) | 40.8 s | 3.4 GB |
| 12 s windows, 2/3 overlap, uniform (authors' layout) | 60.6 s | 3.4 GB |
| CoreML (MLProgram, export with explicit PReLU) | 1034 s | 28.5 GB footprint |

Levers measured and dropped: intra-op threads (4: 50.6 s; 8, 10, 14:
42.6-43.1 s), a batch-2 export (4.79 s per channel-window against
4.13 s unbatched), shorter windows (real-time factor 0.343 / 0.322 /
0.320 per channel at 12 / 6 / 3 s, linear, so no quadratic attention to
save). Profile of one run: Transpose 24.6%, Conv 24.1%,
InstanceNormalization 15.2%, Resize 11.1% of kernel time over about
10,000 nodes.

Sessions and threads (2026-09-26): the same model, mono, no overlap,
the first 144 s (12 windows) of a 1080p60 gameplay recording's audio,
split into N parts at window boundaries and run on N separators at once
(`charon-music-removal bench-par`). Release build, charon-audio
`release/0.1.2` working tree on `6831e48`. Differences against one
session on 14 threads: at most 1.2e-4, the same as one session on 7
threads against 14 (6.6e-5), so they come from the thread count, not
the split. The host has 12 cores: rows with more threads in total
oversubscribe it (the first rows were run before this was noticed; the
1 x 12 and 2 x 6 rows are the ones the app uses).

| sessions x threads | wall | real-time factor | peak RSS |
|---|---|---|---|
| 1 x 14 | 51.2 s | 0.355 | 3.4 GB |
| 1 x 12 | 47.7 s | 0.331 | 3.6 GB |
| 1 x 7 | 53.0 s | 0.368 | 3.6 GB |
| 2 x 7 | 32.7 s | 0.227 | 6.7 GB |
| 2 x 6 | 30.2 s | 0.210 | 7.0 GB |
| 2 x 5 | 33.7 s | 0.234 | 7.0 GB |
| 3 x 4 | 32.0 s | 0.223 | 9.5 GB |
| 4 x 3 | 59.6 s | 0.414 | 12.4 GB |
| 6 x 2 | 266.7 s | 1.852 | 12.7 GB |

One session does not use more than about 6-7 cores on this graph; two
sessions of 6 are 1.58x faster than one of 12. The CPU arena off
changed nothing (3.8 GB, 55.5 s for 1 x 7). onnxslim 0.1.96 left the
graph as it was (250 Transpose, 873 Conv before and after).

The reference PyTorch module itself is pathologically slow on CPU with
NNPACK enabled (97% of the time in `aten::_nnpack_spatial_convolution`,
135,232 calls for 0.5 s of audio); the export script disables NNPACK
for its checks. This concerns only the reference.

# Project summary

charon-audio is a Rust pipeline that runs the HTDemucs music source
separation model through ONNX Runtime: decoding, STFT/iSTFT,
segmentation, overlap-add and output writing in Rust, the network on the
CPU or, on macOS, on the GPU through CoreML. It reproduces demucs 4.1.0
to float precision (max sample difference below 1e-3, equal median SDR
on MUSDB18 previews) and, with its resident server, separates a 193 s
track in 6.2-6.4 s on an Apple M4 Pro, faster than PyTorch on MPS with
the same warm-model treatment.

## What it is for

- A library (`Separator`) for embedding separation in Rust programs
  without Python.
- A CLI (`charon`) with a resident server for batch or service use.
- A measured baseline: every performance and quality claim has a
  record (`docs/MEASUREMENTS.md`) and the scripts to reproduce it
  (`tools/parity/`).

## State (0.1.1)

Supported: HTDemucs 4-stem, three ONNX artifacts (in-graph CPU, split
CPU, split CoreML), WAV/FLAC/MP3/OGG/AAC input, WAV and FLAC output,
resampling, time-shift ensembling, CoreML on macOS, resident server.

Not supported: other model families, CUDA, WebGPU, real-time streaming
(the `realtime` feature is experimental), model download (files are
verified by hash but fetched by the user; see `docs/MODELS.md`).

## Documents

- `README.md`: status, quick start, measurements summary.
- `CHANGELOG.md`: what changed in 0.1.1 and what was removed, with
  reasons.
- `docs/IMPLEMENTATION.md`: the pipeline and its contracts.
- `docs/MODELS.md`: artifacts, hashes, license position, hosting steps.
- `docs/MEASUREMENTS.md`: the numbers and how they were obtained.
- `docs/CONTRIBUTING.md`: acceptance gate and workflow.

# Speed and GPU: where the time goes and which GPU path runs, 2026-09-25

Three questions, each answered with a measurement:
1. Where does CPU time go inside one 7.8 s segment?
2. Does any GPU execution provider execute an HTDemucs graph on this
   machine, and how fast?
3. What does that imply for the pipeline?

## Stamp

- charon commit `80bb09b` plus `examples/ep_probe.rs` (this record's
  tool), `cargo build --release --features ep-experimental --example
  ep_probe`. ort `=2.0.0-rc.13` with the `coreml,webgpu` ONNX Runtime
  1.28.0 build (Dawn on Metal for WebGPU).
- Host: Apple M4 Pro (14 CPU cores, GPU, Neural Engine), 24 GB, macOS 26.6.
- Inputs: random tensors shaped from each model's metadata, so the times
  are per model run, not per track. `max |ep - cpu|` is against a CPU run
  of the same input.
- Models:
  - `htdemucs.onnx` (StemSplitio, SHA-256 `68d0bf16...`): STFT/iSTFT
    inside the graph, input `mix [1,2,343980]`.
  - `HTDemucs-ORT-09dc1655.ort` (gentij/htdemucs-ort, SHA-256
    `09dc1655...bec0729`, 210 MB, ORT format, opset 18): STFT outside
    the graph. Inputs `input [1,2,343980]` (waveform) and `x
    [1,4,2048,336]` (complex-as-channels spectrogram: L.re, L.im, R.re,
    R.im; 2048 bins; 336 frames). Outputs `output [1,4,4,2048,336]`
    (per-source complex spectrogram, iSTFT to be done outside) and
    `add_67 [1,4,2,343980]` (time-branch waveform). The two branches are
    summed after the iSTFT, as in `HTDemucs.forward`.

## 1. CPU profile of one segment, `htdemucs.onnx`

ORT profiler, kernel time of the third run, all 14 threads.

| setting | model_run | top kernels |
|---|---|---|
| ORT defaults (constant folding on) | 608 ms, 1182 nodes | ConvTranspose 156 ms (26%, 10 nodes), Conv 108 (18%), FusedMatMul 67 (11%), Gemm 51 (8%), InstanceNorm 48 (8%), Transpose 36 (6%) |
| ConstantFolding off (the low-memory preset) | 724 ms, 11132 nodes | ConvTranspose 151, Conv 134, **Range 61 ms (696 nodes), ScatterND 38 ms (684), Slice 18 ms (2091)**: 117 ms of STFT/iSTFT index arithmetic |

Reading: the in-graph STFT/iSTFT costs 16% of segment time when its
constants are not folded, and 3.8 GB of memory when they are (memory
record). Moving STFT/iSTFT out of the graph removes both at once. The
network itself (ConvTranspose + Conv + attention) is 85% of the time and
is ORT kernel work; there is little pipeline-level CPU headroom beyond
that 16%.

## 2. Execution providers

Per-run time after warm-up (median of runs 3-5), session build time,
correctness against CPU.

| model | EP | session build | run time | max \|ep - cpu\| | verdict |
|---|---|---|---|---|---|
| htdemucs.onnx | CPU | 1.1 s | 0.61-0.79 s | | baseline |
| htdemucs.onnx | CoreML (4 configs, earlier record) | fails | | | does not run |
| htdemucs.onnx | **WebGPU** | 1.2 s | 0.95 s | 2.5e-5 | **runs, correct, slower than CPU** (index ops fall back to CPU with transfers) |
| .ort (STFT outside) | CPU | 0.1 s | 0.51-0.70 s | | baseline |
| .ort (STFT outside) | **CoreML, All** | 38 s first, 7 s cached | **0.28 s** | 1.6e-4 | **runs, 2.2x CPU** |
| .ort (STFT outside) | CoreML, CPUAndGPU | 28 s | 0.28 s | 1.6e-4 | same as All (the GPU does the work) |
| .ort (STFT outside) | CoreML, CPUAndNeuralEngine | 25 s | 0.41 s | 1.2e-4 | slower than GPU |
| .ort (STFT outside) | WebGPU | 0.1 s | fails | | Softmax kernel error |

The CoreML numbers are what matters: with the STFT outside the graph,
CoreML on the GPU executes the whole network at 0.28 s per segment
against 0.61 s on CPU for the current export. A 193 s track is 33
segments: about 9.2 s of model time plus STFT/iSTFT and I/O, versus
20.4 s today. That lands next to PyTorch on MPS (9.0 s wall, whole
CLI). The 1.6e-4 difference against CPU is the expected fp16/fp32
accumulation difference of the GPU path and is below the 1e-3 parity
criterion; it must be re-checked end to end on MUSDB.

Costs: the first session build compiles the CoreML model (25-38 s);
the compiled model is cached on disk, after which a build takes 7 s.
The cache must live in a persistent directory, not `/tmp`.

## 3. Consequences

**Plan: STFT/iSTFT in Rust, network on ORT with CoreML or CPU.**

- Implement Demucs' `_spec`/`_ispec` exactly (`demucs/htdemucs.py`,
  `demucs/spec.py`): reflect-pad by `hop/2*3` plus alignment to a
  multiple of `hop`, `torch.stft` with a periodic Hann window of 4096,
  hop 1024, `normalized=True` (scale `1/sqrt(4096)`), `center=True`
  with reflect padding, drop the last frequency bin, keep frames
  `[2, 2+le)`; inverse with zero-padded bin and 2-frame pads, overlap-add
  with the squared-window normalization of `torch.istft`, then trim
  `[pad, pad+length)`. Complex-as-channels layout `[B, C*2, F, T]`.
  Then `stems = add_67 + ispec(output)`.
- Use realfft/rustfft; the cost is about 336 FFTs of 4096 per channel
  per segment, a few milliseconds.
- Parity test: STFT/iSTFT round-trip against a `torch.stft` fixture
  (max diff <= 1e-5), and the full pipeline against PyTorch on the
  MUSDB7 set with the same 1e-3 criterion as Q1.
- This also fixes the memory problem at its root (no folded index
  constants), so the low-memory preset and its 11% cost disappear.
- Which export: gentij's `.ort` works today but its contract is
  third-party; the cleaner option is to export our own with
  `demucs-onnx` (StemSplit's exporter), STFT outside, waveform and
  spectrogram outputs, opset 17, and publish its hash. The `.ort` is
  the fallback if the export cannot be reproduced.
- EP selection at run time: CoreML on macOS if the compiled model
  loads, else CPU; every choice measured and printed.

stem-splitter-core already uses this export with CoreML, but its
STFT/iSTFT differs from Demucs (zero-padding of `n_fft/2` instead of
Demucs' reflect padding and frame trimming), which is consistent with
its 0.2-0.3 dB deficit on MUSDB (head-to-head record). Doing the
transform exactly is the whole point.

**Not the plan:**
- WebGPU: runs the current graph but slower than CPU; revisit only
  after the STFT is outside (it then fails on Softmax in the `.ort`,
  which needs an ORT-side fix).
- Neural Engine: slower than the GPU for this network.
- Burn/wgpu reimplementation (demucs-rs' route): weeks of work,
  duplicates an existing crate, and that crate's Metal path produced
  NaN on 6 of 50 tracks here.
- CUDA: unmeasurable on this machine; the same STFT-outside export
  is the prerequisite there too.

**Expected numbers to verify, not to promise:** 193 s track in about
11-13 s wall on the M4 Pro with CoreML (from 26.4 s), peak RSS around
1 GB (from 2.1 GB), CPU-only path 17-18 s (from 20.4-22 s) because the
STFT overhead and the low-memory penalty go away. Effort: 2-3 days
including parity tests.

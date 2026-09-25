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

## 4. Results after implementation (same day)

Stamp: charon commit after `e97ccdb` (the split path), `cargo build
--release --features coreml --example separate --example profile`.
Export: `tools/export/export_htdemucs.py` with demucs 4.1.0, torch
2.14.0, onnx 1.23.0; `htdemucs_split.onnx`, 185,200,998 bytes, SHA-256
`6104c3de08607e0898f70835f1be1ff13a85bbea4826bdba80f9cb7b58fe088f`.
Export-time parity: patched split model vs PyTorch 2.7e-7; ONNX on
onnxruntime 1.30 CPU plus host iSTFT vs PyTorch 6.5e-5.

### STFT/iSTFT in Rust

`src/stft.rs` against `torch.stft`/`torch.istft` fixtures
(`tests/fixtures/make_stft_fixture.py`, 20000 samples): forward and
inverse both within 1e-5 relative to the largest value. Identity-model
pipeline test through the split contract passes at 8192- and
12000-sample segments (`tests/pipeline.rs`).

### Parity against PyTorch demucs 4.1.0 (`shifts=0`)

| input | provider | max \|diff\| (worst stem) | agreement SDR range |
|---|---|---|---|
| 3 synthetic clips (5, 20, 31.3 s) | CPU | 1.2e-4 | 66.8-90.0 dB |
| 3 synthetic clips | CoreML | 1.7e-6 | 81.7-129.2 dB |
| 193 s real track | CPU | 9.2e-4 | 66.5-80.5 dB |
| 193 s real track | CoreML | 1.3e-5 | 114.3-121.3 dB |

The CPU path reproduces the in-graph export's numbers exactly (the same
ORT kernels). The CoreML path is 40 dB closer to PyTorch than ORT's CPU
path; ORT's CPU fused kernels, not charon, account for the CPU gap.
Outputs are finite everywhere.

### Time and memory, 193 s track, M4 Pro

Per-run (`examples/profile`, separation only, run 1 of 2):

| path | load | separate | RTF | RSS after load | peak RSS |
|---|---|---|---|---|---|
| in-graph export, low-memory preset (0.1.1 default before this work) | 2.9 s | 22.8 s | 8.5 | 0.7 GB | 2.1 GB |
| in-graph export, ORT defaults | 1.1 s | 20.4 s | 9.4 | 4.6 GB | 5.6 GB |
| split export, CPU, memory pattern on | 0.2 s | 16.6 s | 11.6 | 0.65 GB | 5.3 GB |
| **split export, CPU, memory pattern off (new CPU default)** | 0.2 s | 17.7 s | 10.9 | 0.65 GB | 3.3 GB |
| **split export, CoreML (GPU)** | 27 s first / 7.4 s cached | **8.2 s** | **23.6** | 2.4 GB | 3.7 GB |

Whole CLI (`separate`, decode + separate + write, `/usr/bin/time -l`,
two passes): split CPU 17.07 / 17.07 s, 3.39 / 3.40 GB; split CoreML
16.17 / 16.20 s, 3.79 / 3.79 GB. For comparison from the head-to-head
record: charon in-graph 26.4 s, PyTorch CPU 35.5 s, PyTorch MPS 9.0 s,
stem-splitter-core 27.7 s, demucs-rs (Metal, NaN output) 15.1 s.

Reading:
- CPU: 20.4 s -> 16.6/17.7 s (the STFT overhead is gone) and the
  low-memory penalty no longer exists; load memory 4.6 GB -> 0.65 GB.
- GPU: separation 8.2 s, 2.5x the CPU path and on par with PyTorch MPS's
  model time. End to end the CoreML CLI is 16.2 s, not 9 s, because
  loading the compiled CoreML model from cache takes 7.4 s of the 16 s.
  That load is one-off per process: a long-running server or a batch of
  tracks pays it once. The remaining lever for a single-track CLI is
  keeping the CoreML session warm (a daemon or a persistent process).
- Peak RSS is now ONNX Runtime's activation memory: with the split graph
  ORT allocates 2.1-3.3 GB during a run (measured with `ep_probe`, no
  charon code: 4.0 GB with memory pattern, 2.7 GB without). The folded
  index constants of the in-graph export are gone, but the planner
  over-allocates for this graph. Next lever: ORT arena configuration
  (`OrtArenaCfg` extend strategy) or running the two branches as
  separate sessions. Not done in this release.

### Quality, MUSDB18 test previews

50 tracks, median whole-signal SDR (dB), `tools/parity/musdb_eval.py`,
PyTorch demucs 4.1.0 `shifts=0` recomputed in the same run.

| path | drums | bass | other | vocals | max \|charon - torch\| |
|---|---|---|---|---|---|
| split export, CoreML | 9.50 | 9.04 | 5.19 | 8.88 | 2.5e-6 |
| split export, CPU | 9.50 | 9.04 | 5.19 | 8.88 | 3.7e-4 |
| in-graph export, CPU (MUSDB7 record) | 9.50 | 9.04 | 5.19 | 8.88 | 3.7e-4 |
| PyTorch demucs 4.1.0 | 9.50 | 9.04 | 5.19 | 8.88 | |

All three charon paths equal the PyTorch reference to two decimals on
every stem; the CoreML path is within 2.5e-6 of it on every track. All
outputs finite.

### What this closes and what remains

Closed: a GPU path on macOS with parity, and a CPU path with the STFT
overhead and the memory/speed trade-off removed.

Open, with numbers to beat:
- End-to-end CoreML CLI is 16.2 s against PyTorch MPS 9.0 s because of
  the 7.4 s compiled-model load; separation itself (8.2 s) is on par.
  A resident process pays the load once.
- Peak RSS 3.3-3.8 GB is ONNX Runtime activation memory on this graph
  (2.7-4.0 GB with no charon code at all). Arena configuration or
  splitting the branches into two sessions are the next experiments.
- The split model is not hosted; the export is reproducible from the
  script and its hash is recorded.


## 5. Second pass: one CoreML partition (same day, later)

The 7 s cached load and the 0.235 s run hid a structural problem, found
by listing the CoreML cache directory: ORT had split the graph into **26
CoreML partitions** with 25 nodes on the CPU in between. The verbose
session log names them: every `Conv`/`ConvTranspose` of the time branch,
rejected with "Input shape {1,2,1,343980} exceeds CoreML convolution
memory limit of 16384". The CoreML provider accepts only 4-D
convolutions with spatial axes of at most 16384; the time branch works
on 343980, 85995 and 21498 samples.

Fix in the export (`tools/export/export_htdemucs.py`): every 1-D
convolution runs as a 2-D convolution with height 1, and when its input
exceeds 16384 samples the time axis is tiled into rows that carry the
kernel's halo, so the 2-D convolution over `[B, C, rows, tile]` equals
the 1-D one exactly (unit-checked at 0.0 for Conv, 3e-6 for
ConvTranspose, which is expressed as zero-stuffing plus a convolution).
The legacy TorchScript exporter mis-traces the tiling (a Concat with
inconsistent shapes), so the tiled export uses the torch.export-based
exporter (`--dynamo`, opset 18, external data folded back into one
file).

Stamp: same host and toolchain; `htdemucs_split_tiled.onnx` (172,439,242
bytes, SHA-256 `782026bd0dbc67e97146271d813f0dc61f5242f80eeefbd6abee5dfe1839607d`),
demucs 4.1.0, torch 2.14.0. Export parity 6.55e-5 through onnxruntime.

### Partition count and per-run time (`ep_probe`, random input)

| export | partitions | nodes on CPU | CoreML run | CPU run | CoreML load (cached) |
|---|---|---|---|---|---|
| split, 1-D convs (section 4) | 26 | 25 | 0.235 s | 0.61 s | 7.4 s |
| split, 2-D convs, untiled (legacy exporter) | 26 | 25 | 0.212 s | 0.45 s | 7.4 s |
| **split, 2-D convs, tiled (dynamo exporter)** | **1** | **0** | **0.157 s** | 0.54 s | 7.0 s |

Other CoreML options on the tiled graph, all measured: compute units
`All` (ANE+GPU) same run time as `CPUAndGPU`; `FastPrediction`
specialization same; `NeuralNetwork` model format catastrophic (190 s
per run, 10 GB). Batch-2 export: 0.267 s per segment on CoreML and
0.84 s on CPU, worse than batch 1, dropped. fp16 weight conversion
(onnxconverter-common) produced an invalid graph and then hung; not
pursued.

The cached load stays at 7 s with one partition, so it is CoreML's
own compile-from-cache of the 2000-op program, not ORT partitioning.
No option exposed by ORT changes it.

### End to end, 193 s track, tiled export on CoreML (two passes)

| | wall | peak RSS | separation only (`profile`) |
|---|---|---|---|
| before (section 4, 26 partitions) | 16.2 s | 3.8 GB | 8.2 s |
| **tiled, one partition** | **13.35 / 13.36 s** | **2.6 GB** | **5.6 s (RTF 34)** |
| PyTorch MPS (head-to-head record) | 9.0 s | 2.0 GB | |

Of the 13.4 s, 7.0 s is the CoreML load, 5.6 s the separation, the rest
decode and write. A resident process is at 5.6 s per 193 s track.

Parity of the tiled CoreML path: synthetic clips max 1.8e-6 (112-128
dB), MUSDB18 previews identical SDR to PyTorch on all stems with max
2.4e-6 per track, all finite.

### CPU artifact

The tiling costs the CPU provider 20% (0.54 vs 0.45 s per run), and the
dynamo exporter's graph is 10% slower on ORT CPU than the legacy
exporter's for the same model (0.498 vs 0.451 s: different
decompositions, fewer fusions). The CPU-target artifact is therefore the
legacy export with 2-D, untiled convolutions (`--no-tiling`):

Measured (`--no-tiling`, legacy exporter, SHA-256 `d5f43acb...`): 0.485 s
per run, CLI 17.7 / 17.6 s, separation 16.5 s, peak 3.7 GB. That is the
original split export's numbers (17.1 s CLI, 16.6 s separation) within
noise, so the 2-D rewrite buys nothing on the CPU provider at pipeline
level. The CPU artifact stays the original export (1-D convolutions,
SHA-256 `6104c3de...`, byte-for-byte reproducible from the script with
`--target cpu`).

### Final artifacts

| artifact | export flags | SHA-256 | size | for |
|---|---|---|---|---|
| `htdemucs_split.onnx` | `--target cpu` (legacy exporter, opset 17) | `6104c3de08607e0898f70835f1be1ff13a85bbea4826bdba80f9cb7b58fe088f` | 185,200,998 | CPU provider |
| `htdemucs_split_coreml.onnx` | `--target coreml` (dynamo exporter, opset 18, tiled 2-D convs) | `782026bd0dbc67e97146271d813f0dc61f5242f80eeefbd6abee5dfe1839607d` | 172,439,242 | CoreML provider |

Both use the same `DemucsSplit` contract and pass the same parity
checks. Reproducibility: `--target cpu` re-exported byte-identical
(same SHA-256). `--target coreml` re-exported with the same 2055 nodes,
757 initializers, identical weights, names and op sequence, but a
different SHA-256 (`30374a0a...`): the dynamo exporter's serialization
is not byte-stable. Verify a re-export structurally, not by hash. Summary of the 193 s track, whole CLI, this host:

| path | wall | peak RSS | separation |
|---|---|---|---|
| in-graph export, CPU (0.1.1 before this work) | 26.4 s | 2.1 GB | 22.8 s |
| split, CPU | 17.1 s | 3.4 GB | 16.6 s |
| split tiled, CoreML | 13.4 s | 2.6 GB | 5.6 s |
| PyTorch CPU / MPS | 35.5 / 9.0 s | 2.3 / 2.0 GB | |

Closed later the same day: the `charon serve` resident server pays the
CoreML load once; jobs then take 6.2-6.4 s per 193 s track
(`2026-09-25-final-head-to-head.md`). Measured about the load itself:
7 s when macOS's compiled-model cache is warm, about 27 s otherwise,
with ORT's on-disk cache entry untouched in both cases (its mtime does
not change), so the recompile is Apple's, not ORT's. Still open:
ONNX Runtime's 2-3 GB activation memory on the CPU provider.


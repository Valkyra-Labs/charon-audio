# Implementation notes

How a file becomes stems, and the contracts each stage keeps. Line
references are to the 0.1.2 sources.

## Pipeline

1. **Decode** (`AudioFile::read`, `src/audio.rs`): Symphonia 0.6 with
   explicit codec features. Samples are taken as decoded, without
   clipping (Symphonia 0.5's MP3 decoder clamped to [-1, 1]; 0.6 does
   not). Decode errors are logged with the failing stage.
2. **Resample and downmix** (`Separator::separate`, `src/separator.rs`):
   to the model's 44.1 kHz stereo. Resampling is rubato's windowed sinc
   in 4096-frame chunks with a flushed tail; impulse tests check that
   the output is time-aligned.
3. **Normalize** (`Processor::process`, `src/processor.rs`): subtract
   the mean and divide by the unbiased standard deviation of the mono
   mix, as `demucs.api.Separator.separate_tensor` does; undo it on the
   stems.
4. **Segment** (`Processor::process_split`): 7.8 s windows (343980
   samples, the model's static input), 25% overlap, stride 5.85 s. Each
   window is centred on its chunk and filled with real neighbouring
   audio where available, zeros elsewhere (`TensorChunk.padded` +
   `center_trim`). Results are blended with the triangular weights of
   `apply_model`, which never reach zero, so no division by zero at the
   edges.
5. **Shifts** (`Processor::process_shifted`, optional): zero-pad half a
   second on both sides, run at evenly spaced offsets, average. Demucs
   draws random offsets; Charon's are deterministic.
6. **Model** (`OnnxModel::infer`, `src/models.rs`), one of three contracts:
   - `Waveform`: `mix [1, 2, 343980]` in, `stems [1, 4, 2, 343980]` out.
     The in-graph export.
   - `DemucsSplit`: `mix` plus the complex-as-channels spectrogram
     `spec [1, 4, 2048, 336]` in; `time [1, 4, 2, 343980]` and
     `spec_out [1, 4, 4, 2048, 336]` out; Charon computes
     `stems = time + ispec(spec_out)`. The spectrogram is
     `HTDemucs._magnitude(_spec(mix))`: reflect-pad by `hop/2*3`, align
     to a multiple of `hop`, `torch.stft` (periodic Hann 4096, hop 1024,
     `normalized=True`, centre reflect padding), drop the Nyquist bin,
     keep frames `[2, 2 + ceil(len/hop))`, layout L.re, L.im, R.re, R.im.
     `_ispec` inverts it: zero Nyquist bin, two zero frames each side,
     `torch.istft` with squared-window normalization, trim
     `[pad, pad + len)`. `src/stft.rs` implements both with realfft
     and is checked against `torch.stft` fixtures at 1e-5 relative.
   - `Spectral` (TIGER-DnR music branch): the model sees one channel's
     complex spectrogram and returns the music stem's. Charon computes
     `torch.stft` with periodic Hann 2048, hop 512, centre reflect
     padding, not normalized (`TorchStft`), and inverts it after the
     model. `per_channel` models run on every channel of any layout, one
     at a time. The TIGER preset uses 12 s windows (529200 samples),
     50% overlap and no normalization; the stride is computed in `f64`
     and rounded, since the model is sensitive to window placement
     (a one-sample drift per window cost 58 dB of agreement).
7. **Write** (`Stems::save_all_as`): WAV 16/24-bit int or 32-bit float,
   FLAC 16/24-bit. flacenc writes the last partial block's size as the
   STREAMINFO minimum block size, which readers take as a
   variable-block stream; `write_flac` sets min = max.

## Progress, cancellation, streaming and regions

- `Control` carries a progress callback, counted in model windows, and
  a `CancelToken`; a cancelled job stops after the current window.
- `Separator::separate_stream` reads an `AudioSource` in blocks and
  writes stems to a `StemSink` in time order, holding a few windows in
  memory. Output is bit-identical to `separate`.
- `Separator::remove_in_regions` runs the model only over the regions
  of a `RegionPlan` plus context, and writes the mix with the target stem
  reduced (`mix - fade * (1 - keep) * estimate`) and the removed part.
  Outside the regions the output is the input bit for bit.

## Execution providers

`OnnxOptions::execution_provider`: `Cpu`, `CoreMl` (feature `coreml`,
macOS), `Auto` (CoreML if the session builds, else CPU with a warning).

CoreML runs the split export only, and runs it as one partition only
when every convolution is 4-D with spatial axes of at most 16384: the
`--target coreml` export runs the time branch's 1-D convolutions as 2-D
ones with the time axis tiled into rows with kernel halos (exact). The
compiled CoreML model is cached under `coreml-cache/<sha256 prefix>/`
next to the model, because ONNX Runtime keys its cache by path and would
load a stale compiled model after a re-export. macOS additionally
compiles the cached program on load: about 7 s when its own cache is
warm, about 27 s otherwise. `charon serve` keeps the session resident to
pay that once.

ONNX Runtime session options that matter, all measured:
- In-graph export: `ConstantFolding` off and memory pattern off
  (`OnnxOptions::low_memory()`), because folding materializes 3.8 GB of
  STFT index constants.
- Split export on CPU: graph optimization level `Extended` (level 3's
  layout transforms cost 4% on Apple Silicon), memory pattern off
  (2 GB of peak for 6% of time).

## Threads and memory

Segments run one at a time; ONNX Runtime uses all cores inside a
segment (`intra_threads`), and the STFT/iSTFT of the channels and
sources run on rayon. Peak RSS is ONNX Runtime's activation memory on
this graph (2-3 GB for HTDemucs, about 3.4 GB for TIGER); the audio
itself is about 1 MB per second of input across all stems.

A separator runs one window at a time. TIGER does not get faster past
about 7 intra-op threads, so an application with the cores and memory
for it runs two separators (two `Separator::new`, two sessions) on
independent parts of the input: channels, or spans that start on a
window boundary when there is no overlap. `with_process_config` gives
another separator on the same session, which saves loading but not
time, since runs on one session take turns.

## Resident server

`charon serve` binds a Unix socket and answers one JSON object per line:
`{"kind":"Separate","input":...,"output_dir":...,"format":...,"shifts":N}`,
`{"kind":"Ping"}`, `{"kind":"Stop"}`. Replies carry `ok`, `stems`,
`seconds` and `provider`. Jobs run sequentially in the server process;
the client only waits.

## What is deliberately not here

Candle and WASM backends (removed: placeholders that did not build),
GPU features that only forwarded build flags (removed), the
`performance` module (deprecated, unused by the pipeline), CUDA
(unmeasured, no hardware), model families other than HTDemucs and the
TIGER-DnR music branch (each needs its own contract and parity record).

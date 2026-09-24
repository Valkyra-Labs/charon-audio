# HTDemucs on full-length real tracks, 2026-09-24

Three full-length MP3 tracks (cinematic/trailer music, 110-193 s,
44.1 kHz stereo) supplied by the owner. The tracks are not in the repo.
They have no ground-truth stems, so **separation quality (SDR) cannot be
measured**. This record covers parity, stability, and the decoding path.

## Stamp

- charon: commit `1793d73`, `cargo build --release --example separate`,
  default features. ort `=2.0.0-rc.13` (ONNX Runtime 1.28.0), CPU EP.
- Model: `htdemucs.onnx`, SHA-256
  `68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74`.
- Reference: demucs 4.1.0, torch 2.14.0, CPU, `shifts=0`.
- Decoders: Symphonia 0.5.5 (`symphonia-bundle-mp3` 0.5.5) in charon;
  ffmpeg (Homebrew) for the reference WAV.
- Script: `tools/parity/unlabeled_eval.py`. The output was reproduced
  identically on a second run. Host: Apple M4 Pro, 24 GB.

## Results

| track | length | max \|charon - torch\| (worst stem) | min agreement vs torch | sum(stems) vs mix | finite |
|---|---|---|---|---|---|
| audioknap-blockbuster | 192.6 s | 9.28e-4 (bass) | 66.5 dB | 28.9 dB | yes |
| hitslab-adventure | 131.5 s | 1.91e-4 (other) | 66.0 dB | 29.2 dB | yes |
| mondamusic-shorts | 109.9 s | 3.96e-5 (bass) | 67.0 dB | 38.9 dB | yes |

- Parity with PyTorch holds on full-length audio, on the same
  ffmpeg-decoded input. Every stem is within the Q1 criterion of 1e-3,
  but the worst case (9.28e-4) is close to it. The criterion was set on
  short clips and may need restating per sample count.
- Sum of stems vs mix: 29-39 dB. HTDemucs is not mixture-consistent by
  construction. This is model behaviour, not a charon defect.

## Finding: Symphonia clamps MP3 output to [-1, 1]

`symphonia-bundle-mp3-0.5.5/src/synthesis.rs:329` has
`*o = s.clamp(-1.0, 1.0)`. Two of the three tracks decode above full
scale (ffmpeg max |x| 1.0905 and 1.0494; 154 and 618 samples above 1).
Symphonia hard-clips these. With the identity model, which writes out
Symphonia's decoded audio, every sample where the two decoders differ by
more than 1e-3 is one of those over-full-scale samples (142/142 and
542/542). On the track without overs, the decoders agree to 1.86e-6.

Effect on stems: charon's stems from the MP3 agree with its stems from
the ffmpeg WAV at 34.5-59.6 dB on the affected tracks, versus 76.7-121.4
dB on the unaffected one. The clamp is in the upstream decoder, not in
charon. Options: report upstream, patch the decoder, or accept it and
document it.

## Timing (single run, not a benchmark)

`/usr/bin/time -l` on the example binary, including model load and MP3
decode, run once on the first pass (commit `1793d73`):

| track | wall | peak RSS |
|---|---|---|
| audioknap-blockbuster (192.6 s) | 24.2 s | 5.49 GB |
| hitslab-adventure (131.5 s) | 16.5 s | 5.66 GB |
| mondamusic-shorts (109.9 s) | 13.9 s | 5.61 GB |

PyTorch separation of the first track took 33.5 s. That figure excludes
model load and decoding, so it is not comparable with charon's wall time,
and no speed claim follows from it.

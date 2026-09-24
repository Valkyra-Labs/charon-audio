# HTDemucs on MUSDB18 7-second previews, 2026-09-24

Closes the M5 acceptance criterion "real music is separated". It
complements Q1 (`2026-09-24-htdemucs-q1.md`), which used synthetic input.

## Stamp

- charon: commit `b921861`, `cargo build --release --example separate`,
  default features. ort `=2.0.0-rc.13` (ONNX Runtime 1.28.0), CPU EP.
- Model: `htdemucs.onnx` (StemSplitio/htdemucs-onnx), SHA-256
  `68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74`.
- Reference: demucs 4.1.0, torch 2.14.0, CPU, `Separator("htdemucs",
  shifts=0, overlap=0.25, split=True)`, applied to the same decoded
  mixture.
- Data: `MUSDB18-7-STEMS.zip` (sigsep-mus-db release v0.4.0,
  147,209,385 bytes), **test split only, 50 tracks**. HTDemucs was
  trained on the train split, so the train split is excluded. Audio was
  decoded by ffmpeg from AAC `.stem.mp4`, so the stems are AAC-coded, not
  HQ. License: non-commercial / educational. No audio is committed.
- Script: `tools/parity/musdb_eval.py`. Host: Apple M4 Pro.

## Metric

**Whole-signal SDR** per stem and track: `10*log10(sum(s^2) / sum((s - s_hat)^2))`,
over both channels, median over tracks. This is **not** museval BSSEval
v4: there are no distortion filters and no 1 s framing. The numbers are
therefore not comparable with published MUSDB tables (for example the
9.0 dB that the demucs README reports for htdemucs). The inputs are also
7 s AAC excerpts, not full HQ tracks.

## Result

| stem | tracks | charon median SDR (dB) | PyTorch demucs median SDR (dB) | max \|charon - torch\| | mixture-as-estimate median SDR (dB) |
|---|---|---|---|---|---|
| drums | 50 | 9.50 | 9.50 | 3.04e-4 | -4.13 |
| bass | 50 | 9.04 | 9.04 | 3.65e-4 | -6.89 |
| other | 50 | 5.19 | 5.19 | 1.50e-4 | -4.98 |
| vocals | 50 | 8.88 | 8.88 | 8.52e-5 | -4.49 |

- charon separates real music. Every stem improves by 10 to 16 dB over
  the mixture-as-estimate baseline, which is what the 0.1.0 identity
  placeholder produced.
- charon's quality equals the PyTorch reference to two decimals on every
  stem. The largest sample difference is 3.65e-4, within the Q1 1e-3
  criterion.
- No listening test was done. The quality evidence is the SDR above.

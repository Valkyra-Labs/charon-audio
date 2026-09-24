# Parity harness

Checks that charon produces the same stems as the reference
implementations for the same model weights.

Requirements: Python 3.12 with `demucs==4.1.0 soundfile onnxruntime numpy`
(for example `uv venv && uv pip install ...`), plus `htdemucs.onnx` from
`https://huggingface.co/StemSplitio/htdemucs-onnx`
(SHA-256 `68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74`).

```bash
python make_mix.py                       # synthetic mix_{5,20,31.3}s.wav
for m in mix_5s mix_20s mix_31.3s; do
  ../../target/release/examples/separate $m.wav charon/$m htdemucs.onnx
done
python reference.py onnx htdemucs.onnx mix_*.wav   # numpy + onnxruntime port
python reference.py torch - mix_*.wav              # PyTorch demucs 4.1.0
python compare.py charon ref_onnx
python compare.py charon ref_torch
```

`compare.py` reports max |diff| and an agreement SDR per stem. The
agreement SDR measures how closely two pipelines agree. It is not a
measure of separation quality.

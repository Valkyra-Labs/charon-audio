"""Reference stems for parity checks.

  onnx:  numpy + onnxruntime port of demucs apply_model(shifts=0, split=True,
         overlap=0.25) + Separator normalization, on the same htdemucs.onnx.
  torch: demucs 4.1.0 PyTorch, demucs.api.Separator('htdemucs', shifts=0).
Writes ref_<kind>/<mix>/<stem>.wav (float32)."""
import sys, pathlib, numpy as np, soundfile as sf
kind, model_path, mixes = sys.argv[1], sys.argv[2], sys.argv[3:]
SOURCES = ("drums", "bass", "other", "vocals")
SEG = 343980

def onnx_separate(sess, wav):
    ref = wav.mean(0); mean = ref.mean(); std = ref.std(ddof=1) + 1e-8
    x = (wav - mean) / std
    C, L = x.shape
    stride = int(0.75 * SEG)
    w = np.concatenate([np.arange(1, SEG // 2 + 1), np.arange(SEG - SEG // 2, 0, -1)]).astype(np.float32)
    w /= w.max()
    out = np.zeros((4, C, L), np.float32); sw = np.zeros(L, np.float32)
    for off in range(0, L, stride):
        clen = min(SEG, L - off); delta = SEG - clen
        start = off - delta // 2; end = start + SEG
        cs, ce = max(0, start), min(L, end)
        win = np.zeros((C, SEG), np.float32); win[:, cs - start:cs - start + ce - cs] = x[:, cs:ce]
        y = sess.run(["stems"], {"mix": win[None]})[0][0]
        y = y[..., delta // 2: delta // 2 + clen]
        out[..., off:off + clen] += y * w[:clen]; sw[off:off + clen] += w[:clen]
    out /= sw
    return out * std + mean

if kind == "onnx":
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    run = lambda wav: onnx_separate(sess, wav)
else:
    import torch, demucs.api
    torch.manual_seed(0)
    sep = demucs.api.Separator(model="htdemucs", shifts=0, overlap=0.25, split=True, device="cpu")
    assert tuple(sep.model.sources) == SOURCES, sep.model.sources
    run = lambda wav: np.stack([s.numpy() for s in sep.separate_tensor(torch.from_numpy(wav), 44100)[1].values()])

for m in mixes:
    wav, sr = sf.read(m, dtype="float32", always_2d=True); assert sr == 44100
    out = run(np.ascontiguousarray(wav.T))
    d = pathlib.Path(f"ref_{kind}") / pathlib.Path(m).stem; d.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(SOURCES):
        sf.write(d / f"{s}.wav", out[i].T, 44100, subtype="FLOAT")
    print(kind, m, "done")

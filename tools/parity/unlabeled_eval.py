"""Checks on real tracks that have no ground-truth stems.

Usage: python unlabeled_eval.py <charon separate binary> <htdemucs.onnx> <identity_4stems.onnx> <work dir> <track>...

Per track:
  1. charon on the original file (product path, Symphonia decoding);
  2. charon and PyTorch demucs 4.1.0 ('htdemucs', shifts=0) on the same
     ffmpeg-decoded float WAV: pipeline parity on full-length audio;
  3. charon with the identity model on the original file, which writes
     Symphonia's decoded audio, compared with ffmpeg's decode: decoder check.

Reports max |charon - torch| per stem, agreement SDR
10*log10(sum(ref^2) / sum((ref - est)^2)) (pipeline agreement, NOT
separation quality), sum-of-stems vs mix agreement, and decoder
differences. No separation-quality metric is possible without reference
stems."""
import subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, demucs.api

SOURCES = ("drums", "bass", "other", "vocals")
charon, model, identity, work = map(Path, sys.argv[1:5])
tracks = [Path(t) for t in sys.argv[5:]]
sep = demucs.api.Separator(model="htdemucs", shifts=0, overlap=0.25, split=True, device="cpu")
rd = lambda p: sf.read(p, dtype="float32", always_2d=True)[0]
agree = lambda r, e: 10 * np.log10((r ** 2).sum() / max(((r - e) ** 2).sum(), 1e-20))
run = lambda inp, out, m: subprocess.run([str(charon), str(inp), str(out), str(m)], check=True,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for track in tracks:
    d = work / track.stem
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(track), "-c:a", "pcm_f32le",
                    str(d / "mix.wav")], check=True)
    run(track, d / "charon_orig", model)
    run(d / "mix.wav", d / "charon_wav", model)
    run(track, d / "decoded", identity)

    mix = rd(d / "mix.wav")
    _, ref = sep.separate_tensor(torch.from_numpy(np.ascontiguousarray(mix.T)), 44100)
    wav = {s: rd(d / "charon_wav" / f"{s}.wav") for s in SOURCES}
    orig = {s: rd(d / "charon_orig" / f"{s}.wav") for s in SOURCES}
    dec = rd(d / "decoded" / "drums.wav")

    print(f"== {track.name}: {len(mix) / 44100:.1f} s")
    print(f"   finite (original-file path): {all(np.isfinite(v).all() for v in orig.values())}")
    print(f"   sum(stems) vs mix agreement: {agree(mix, sum(wav.values())):.1f} dB")
    diff = np.abs(mix - dec)
    over = np.abs(mix) > 1
    big = diff > 1e-3
    print(f"   decoder: ffmpeg max|x| {np.abs(mix).max():.4f} (samples >1: {over.sum()}), "
          f"symphonia max|x| {np.abs(dec).max():.4f}; max|diff| {diff.max():.2e}; "
          f"samples diff>1e-3: {big.sum()}, of which ffmpeg |x|>1: {(big & over).sum()}")
    for s in SOURCES:
        e, r = wav[s], ref[s].numpy().T
        print(f"   {s:7} rms {np.sqrt((e ** 2).mean()):.4f}  charon-vs-torch max|d| {np.abs(e - r).max():.2e} "
              f"agree {agree(r, e):5.1f} dB  original-vs-wav agree {agree(e, orig[s]):5.1f} dB")

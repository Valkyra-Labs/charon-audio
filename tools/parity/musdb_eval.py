"""Separation quality on the MUSDB18 7-second preview set.

Usage: python musdb_eval.py <MUSDB18-7 dir> <charon separate binary> <htdemucs.onnx> <work dir>

For every .stem.mp4 track: ffmpeg extracts mixture and ground-truth stems
(stream order: mixture, drums, bass, other, vocals) to float WAV; charon
and PyTorch demucs 4.1.0 ('htdemucs', shifts=0) separate the same decoded
mixture.

Metric: whole-signal SDR = 10*log10(sum(s^2) / sum((s - s_hat)^2)) per
stem and track, summed over both channels. This is NOT museval BSSEval v4
(no distortion filters, no 1 s framing), so the numbers are not comparable
with published MUSDB SDR tables. Tracks where a ground-truth stem is
silent (energy < 1e-8 per sample) are skipped for that stem. Aggregation:
median over tracks.

The MUSDB18 license is non-commercial / educational: do not commit the
audio or the separated stems."""
import subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, demucs.api

SOURCES = ("drums", "bass", "other", "vocals")
root, charon_bin, model, work = map(Path, sys.argv[1:5])
work.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
sep = demucs.api.Separator(model="htdemucs", shifts=0, overlap=0.25, split=True, device="cpu")

def sdr(ref, est):
    return 10 * np.log10((ref ** 2).sum() / max(((ref - est) ** 2).sum(), 1e-20))

rows = []
for track in sorted(root.rglob("*.stem.mp4")):
    name = track.name.removesuffix(".stem.mp4")
    tdir = work / name
    tdir.mkdir(exist_ok=True)
    for idx, stem in enumerate(("mixture",) + SOURCES):
        out = tdir / f"{stem}.wav"
        if not out.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(track), "-map", f"0:a:{idx}",
                            "-ar", "44100", "-c:a", "pcm_f32le", str(out)], check=True)
    mix, sr = sf.read(tdir / "mixture.wav", dtype="float32", always_2d=True)
    subprocess.run([str(charon_bin), str(tdir / "mixture.wav"), str(tdir / "charon"), str(model)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, torch_out = sep.separate_tensor(torch.from_numpy(np.ascontiguousarray(mix.T)), sr)
    for s in SOURCES:
        ref, _ = sf.read(tdir / f"{s}.wav", dtype="float32", always_2d=True)
        est, _ = sf.read(tdir / "charon" / f"{s}.wav", dtype="float32", always_2d=True)
        tor = torch_out[s].numpy().T
        n = min(len(ref), len(est), len(tor))
        ref, est, tor = ref[:n], est[:n], tor[:n]
        silent = (ref ** 2).mean() < 1e-8
        rows.append((name, s, None if silent else sdr(ref, est), None if silent else sdr(ref, tor),
                     np.abs(est - tor).max()))
    print(f"done {name}", flush=True)

print(f"\n{'stem':7} {'tracks':>6} {'charon_sdr_med':>14} {'torch_sdr_med':>13} {'max|charon-torch|':>18}")
for s in SOURCES:
    r = [x for x in rows if x[1] == s]
    scored = [x for x in r if x[2] is not None]
    print(f"{s:7} {len(scored):6d} {np.median([x[2] for x in scored]):14.2f} "
          f"{np.median([x[3] for x in scored]):13.2f} {max(x[4] for x in r):18.2e}")
print(f"\ntracks: {len({x[0] for x in rows})}")

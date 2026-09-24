"""Separation quality of an external tool on the MUSDB18 7-second previews,
with the same metric as musdb_eval.py (whole-signal SDR, median over
tracks, silent reference stems skipped).

Usage: python musdb_eval_tool.py <MUSDB18-7 dir> <work dir> <layout> <command...>

The command is run per track with {in} and {out} substituted. Layouts:
  stem_splitter   <out>/mixture_<stem>.wav
  demucs_rs       <out>/<stem>.wav
  charon          <out>/<stem>.wav
Non-finite samples in a tool's output are counted and treated as zeros
for the SDR, and reported."""
import subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf

SOURCES = ("drums", "bass", "other", "vocals")
root, work, layout = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cmd = sys.argv[4:]
work.mkdir(parents=True, exist_ok=True)
path_of = {
    "stem_splitter": lambda out, s: out / f"mixture_{s}.wav",
    "demucs_rs": lambda out, s: out / f"{s}.wav",
    "charon": lambda out, s: out / f"{s}.wav",
}[layout]

def sdr(ref, est):
    return 10 * np.log10((ref ** 2).sum() / max(((ref - est) ** 2).sum(), 1e-20))

rows, nonfinite_tracks = [], 0
for track in sorted(root.rglob("*.stem.mp4")):
    name = track.name.removesuffix(".stem.mp4")
    tdir = work / name
    tdir.mkdir(exist_ok=True)
    for idx, stem in enumerate(("mixture",) + SOURCES):
        out = tdir / f"{stem}.wav"
        if not out.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(track), "-map", f"0:a:{idx}",
                            "-ar", "44100", "-c:a", "pcm_f32le", str(out)], check=True)
    out_dir = tdir / layout
    out_dir.mkdir(exist_ok=True)
    argv = [a.replace("{in}", str(tdir / "mixture.wav")).replace("{out}", str(out_dir)) for a in cmd]
    subprocess.run(argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    track_bad = 0
    for s in SOURCES:
        ref, _ = sf.read(tdir / f"{s}.wav", dtype="float32", always_2d=True)
        est, _ = sf.read(path_of(out_dir, s), dtype="float32", always_2d=True)
        n = min(len(ref), len(est))
        ref, est = ref[:n], est[:n]
        bad = ~np.isfinite(est)
        track_bad += bad.sum()
        est = np.where(bad, 0.0, est)
        silent = (ref ** 2).mean() < 1e-8
        rows.append((name, s, None if silent else sdr(ref, est)))
    nonfinite_tracks += track_bad > 0
    print(f"done {name}" + (f"  non-finite samples: {track_bad}" if track_bad else ""), flush=True)

print(f"\n{layout}: {len({r[0] for r in rows})} tracks, {nonfinite_tracks} with non-finite output")
print(f"{'stem':7} {'tracks':>6} {'median_sdr':>10}")
for s in SOURCES:
    v = [r[2] for r in rows if r[1] == s and r[2] is not None]
    print(f"{s:7} {len(v):6d} {np.median(v):10.2f}")

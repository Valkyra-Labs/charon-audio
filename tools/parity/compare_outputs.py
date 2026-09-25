"""Agreement of each tool's stems with PyTorch demucs (CPU) on one track.

Usage: python compare_outputs.py <h2h dir> <track stem name> [ref subdir, default torch_ref]

The reference must be written with `--float32 --clip-mode none`; the
demucs CLI default (int16, clip-mode rescale) rescales loud stems.

Layouts: charon_*/<stem>.wav, stem_splitter/<name>_<stem>.wav,
demucs_rs/<stem>.wav, torch_*/htdemucs/<name>/<stem>.wav.
agree_sdr = 10*log10(sum(ref^2)/sum((ref-est)^2)) over both channels,
after trimming to the common length. Pipeline agreement, not quality."""
import sys
from pathlib import Path
import numpy as np, soundfile as sf

root, name = Path(sys.argv[1]), sys.argv[2]
ref_dir = sys.argv[3] if len(sys.argv) > 3 else "torch_ref"
SOURCES = ("drums", "bass", "other", "vocals")
layouts = {
    "charon_lowmem": lambda s: root / "charon_lowmem" / f"{s}.wav",
    "charon_maxspeed": lambda s: root / "charon_maxspeed" / f"{s}.wav",
    "stem_splitter": lambda s: root / "stem_splitter" / f"{name}_{s}.wav",
    "demucs_rs": lambda s: root / "demucs_rs" / f"{s}.wav",
    "torch_cpu": lambda s: root / "torch_cpu" / "htdemucs" / name / f"{s}.wav",
    "torch_mps": lambda s: root / "torch_mps" / "htdemucs" / name / f"{s}.wav",
}
ref_of = lambda s: root / ref_dir / "htdemucs" / name / f"{s}.wav"
rd = lambda p: sf.read(p, dtype="float32", always_2d=True)[0]

print(f"{'tool':16} " + " ".join(f"{s:>14}" for s in SOURCES) + f"   (agreement SDR dB vs {ref_dir}; max|diff|)")
for tool, path in layouts.items():
    cells = []
    for s in SOURCES:
        try:
            ref, est = rd(ref_of(s)), rd(path(s))
        except Exception as e:
            cells.append(f"{'missing':>14}")
            continue
        n = min(len(ref), len(est))
        if abs(len(ref) - len(est)) > 4410:
            cells.append(f"len {len(est)}/{len(ref)}")
            continue
        ref, est = ref[:n], est[:n]
        if not np.isfinite(est).all():
            cells.append(f"NaN x{(~np.isfinite(est)).sum()}")
            continue
        sdr = 10 * np.log10((ref ** 2).sum() / max(((ref - est) ** 2).sum(), 1e-20))
        cells.append(f"{sdr:6.1f} {np.abs(ref - est).max():7.1e}")
    print(f"{tool:16} " + " ".join(cells))

"""Agreement between charon stems and a reference pipeline.
agree_sdr = 10*log10(sum(ref^2) / sum((ref-est)^2)) over the whole signal,
per stem. It measures pipeline agreement, NOT separation quality."""
import sys, pathlib, numpy as np, soundfile as sf
est_root, ref_root = map(pathlib.Path, sys.argv[1:3])
print(f"{'mix':10} {'stem':7} {'max|diff|':>10} {'agree_sdr_dB':>12}")
for mixdir in sorted(ref_root.iterdir()):
    for s in ("drums", "bass", "other", "vocals"):
        r, _ = sf.read(mixdir / f"{s}.wav", dtype="float32")
        e, _ = sf.read(est_root / mixdir.name / f"{s}.wav", dtype="float32")
        assert r.shape == e.shape, (r.shape, e.shape)
        assert np.isfinite(e).all(), "non-finite in charon output"
        err = r - e
        sdr = 10 * np.log10((r ** 2).sum() / max((err ** 2).sum(), 1e-20))
        print(f"{mixdir.name:10} {s:7} {np.abs(err).max():10.2e} {sdr:12.1f}")

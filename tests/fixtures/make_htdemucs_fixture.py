"""Generate the fixture for the ignored real-model test in tests/htdemucs.rs.

  synth_9s.flac                 9 s synthetic stereo mix, 16-bit FLAC
  synth_9s_htdemucs_rms.json    per-stem, per-channel RMS over 50 ms frames
                                of PyTorch demucs 4.1.0 'htdemucs'
                                (Separator shifts=0, overlap=0.25, split=True)

Requires demucs==4.1.0, torch, soundfile, numpy. Run from this directory."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, demucs, demucs.api

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "parity"))
from make_mix import mix  # noqa: E402

FRAME = 2205  # 50 ms at 44.1 kHz
sf.write("synth_9s.flac", mix(9.0, 90), 44100, subtype="PCM_16")
wav, sr = sf.read("synth_9s.flac", dtype="float32", always_2d=True)

sep = demucs.api.Separator(model="htdemucs", shifts=0, overlap=0.25, split=True, device="cpu")
_, stems = sep.separate_tensor(torch.from_numpy(np.ascontiguousarray(wav.T)), sr)

n_frames = wav.shape[0] // FRAME
rms = {
    name: [
        [float(np.sqrt(np.mean(ch[i * FRAME:(i + 1) * FRAME] ** 2))) for i in range(n_frames)]
        for ch in stem.numpy()
    ]
    for name, stem in stems.items()
}
meta = {
    "reference": f"demucs {demucs.__version__}, torch {torch.__version__}, htdemucs, shifts=0",
    "frame_samples": FRAME,
    "sources": list(stems.keys()),
    "rms": rms,
}
Path("synth_9s_htdemucs_rms.json").write_text(json.dumps(meta, indent=None))
print("frames", n_frames, "sources", list(stems.keys()))

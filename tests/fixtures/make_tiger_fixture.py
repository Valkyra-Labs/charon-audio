"""Fixture for the ignored TIGER test in tests/htdemucs.rs.

  synth_9s_tiger_rms.json   per-channel RMS over 50 ms frames of the music
                            source of TIGER-DnR's music network (PyTorch)
                            on synth_9s.flac

The input is shorter than one 12 s window, so each channel is run as one
window with the audio centred and zeros around it, exactly as Charon's
processor places a short input (`Processor::run_window`).

Usage: make_tiger_fixture.py TIGER_REPO WEIGHTS_DIR (run from this directory)."""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "export"))
import export_tiger  # noqa: E402

FRAME = 2205
WINDOW = 529_200
torch.backends.nnpack.set_flags(False)
model, cfg = export_tiger.load(Path(sys.argv[1]), Path(sys.argv[2]))
wav, sr = sf.read("synth_9s.flac", dtype="float32", always_2d=True)
assert sr == 44100
n = wav.shape[0]
left = (WINDOW - n) // 2
rms = []
with torch.no_grad():
    for ch in wav.T:
        x = np.zeros(WINDOW, dtype=np.float32)
        x[left:left + n] = ch
        music = model.music(torch.from_numpy(x)[None])[0, 0].numpy()[left:left + n]
        rms.append([float(np.sqrt(np.mean(music[i * FRAME:(i + 1) * FRAME] ** 2)))
                    for i in range(n // FRAME)])
meta = {
    "reference": f"TIGER-DnR music branch, torch {torch.__version__}, one centred 12 s window",
    "frame_samples": FRAME,
    "sources": ["music"],
    "rms": {"music": rms},
}
Path("synth_9s_tiger_rms.json").write_text(json.dumps(meta))
print("frames", n // FRAME)

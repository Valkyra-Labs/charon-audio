"""PyTorch demucs 4.1.0 with the model resident: time consecutive
separations of one file on a device, including decode and WAV writing,
the same work as `charon serve` does per job.

Usage: python torch_resident.py <track.wav> <out dir> <device> [runs]"""
import sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, demucs.api

track, out, device = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
t = time.perf_counter()
sep = demucs.api.Separator(model="htdemucs", shifts=0, overlap=0.25, split=True, device=device)
print(f"model loaded on {device} in {time.perf_counter() - t:.2f} s")
out.mkdir(parents=True, exist_ok=True)
for i in range(runs):
    t = time.perf_counter()
    wav, sr = sf.read(track, dtype="float32", always_2d=True)
    _, stems = sep.separate_tensor(torch.from_numpy(np.ascontiguousarray(wav.T)), sr)
    for name, s in stems.items():
        sf.write(out / f"{name}.wav", s.cpu().numpy().T, sr, subtype="FLOAT")
    print(f"run {i}: {time.perf_counter() - t:.2f} s (decode + separate + write)")

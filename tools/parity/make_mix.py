"""Synthetic stereo test mixes (float32 WAV, 44.1 kHz): kick/noise drums,
sine bass, vibrato 'voice' with harmonics, pad chords. Deterministic."""
import sys, numpy as np, soundfile as sf
SR = 44100
def mix(seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    beat = (t * 2.0) % 1.0
    kick = np.sin(2 * np.pi * 55 * t * (1 + 2 * np.exp(-beat * 30))) * np.exp(-beat * 12)
    hat = rng.standard_normal(t.size) * np.exp(-((t * 4.0) % 1.0) * 60) * 0.3
    bass = 0.5 * np.sin(2 * np.pi * np.where((t % 4) < 2, 55, 73.4) * t)
    f0 = 220 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t)) * np.where((t % 3) < 1.5, 1, 1.26)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 8)) * 0.3 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))
    pad = 0.15 * sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    left = kick + hat + bass + 0.8 * voice + 1.2 * pad
    right = kick + 0.7 * hat + bass + 1.2 * voice + 0.8 * pad
    x = np.stack([left, right], axis=1)
    return (x / np.abs(x).max() * 0.8).astype(np.float32)
if __name__ == "__main__":
    for seconds in (5.0, 20.0, 31.3):
        sf.write(f"mix_{seconds:g}s.wav", mix(seconds, int(seconds * 10)), SR, subtype="FLOAT")
    print("ok")

"""STFT/iSTFT parity fixtures for the Rust implementation of Demucs'
HTDemucs._spec / _ispec (demucs 4.1.0, htdemucs.py, spec.py).

Writes little-endian f32 raw files:
  stft_in.f32        [1 ch, N] input
  stft_spec.f32      [2, F, T] complex-as-channels (re, im) output of _spec
  istft_spec.f32     [2, F, T] random spectrogram
  istft_out.f32      [N] output of _ispec(istft_spec, N)
and stft_fixture.json with N, F, T, n_fft, hop.
Run from this directory with demucs 4.1.0 installed."""
import json, math, numpy as np, torch, torch.nn.functional as F
from demucs.spec import spectro, ispectro

N_FFT, HOP, N = 4096, 1024, 20000
torch.manual_seed(0)

def _spec(x):
    le = math.ceil(x.shape[-1] / HOP)
    pad = HOP // 2 * 3
    x = F.pad(x, (pad, pad + le * HOP - x.shape[-1]), mode="reflect")
    z = spectro(x, N_FFT, HOP)[..., :-1, :]
    assert z.shape[-1] == le + 4
    return z[..., 2: 2 + le]

def _ispec(z, length):
    z = F.pad(z, (0, 0, 0, 1))
    z = F.pad(z, (2, 2))
    pad = HOP // 2 * 3
    le = HOP * math.ceil(length / HOP) + 2 * pad
    x = ispectro(z, HOP, length=le)
    return x[..., pad: pad + length]

t = torch.arange(N, dtype=torch.float32) / 44100
x = (0.5 * torch.sin(2 * math.pi * 220 * t) + 0.3 * torch.sin(2 * math.pi * 3001 * t)
     + 0.1 * torch.randn(N))[None]
z = _spec(x)                      # (1, F, T) complex
spec = torch.view_as_real(z[0]).permute(2, 0, 1).contiguous()  # (2, F, T)
Fq, T = z.shape[-2:]

zr = torch.randn(1, Fq, T, dtype=torch.complex64) * 0.1
y = _ispec(zr, N)
istft_spec = torch.view_as_real(zr[0]).permute(2, 0, 1).contiguous()

x[0].numpy().astype("<f4").tofile("stft_in.f32")
spec.numpy().astype("<f4").tofile("stft_spec.f32")
istft_spec.numpy().astype("<f4").tofile("istft_spec.f32")
y[0].numpy().astype("<f4").tofile("istft_out.f32")
# round trip of the real signal through _spec -> _ispec, for the tolerance
rt = _ispec(z, N)
json.dump({"n": N, "freqs": Fq, "frames": T, "n_fft": N_FFT, "hop": HOP,
           "roundtrip_max_err": float((rt - x).abs().max())}, open("stft_fixture.json", "w"))
print("F", Fq, "T", T, "roundtrip max err", float((rt - x).abs().max()))

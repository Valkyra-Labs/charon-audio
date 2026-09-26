#!/usr/bin/env python3
"""Fixtures for `TorchStft` (src/stft.rs): plain torch.stft / torch.istft
with n_fft 2048, hop 512, periodic Hann, center=True, reflect padding,
normalized=False, onesided (TIGER's transform).

Writes little-endian f32 files:
- torch_stft_in.f32: 5000-sample signal
- torch_stft_spec.f32: its spectrum as [2 (re, im), 1025, frames]
- torch_istft_spec.f32: a modified spectrum (scaled and phase-shifted)
- torch_istft_out.f32: torch.istft of it with length=5000
"""
from pathlib import Path

import numpy as np
import torch

N_FFT, HOP, LEN = 2048, 512, 5000
out = Path(__file__).parent
torch.manual_seed(1234)
x = 0.3 * torch.randn(LEN) + 0.2 * torch.sin(torch.arange(LEN) * 0.05)
win = torch.hann_window(N_FFT)
spec = torch.stft(x, N_FFT, HOP, window=win, return_complex=True)
ri = torch.stack([spec.real, spec.imag]).numpy().astype("<f4")
mod = spec * (0.5 + torch.rand(spec.shape)) * torch.exp(1j * torch.rand(spec.shape))
y = torch.istft(mod, N_FFT, HOP, window=win, length=LEN)
mod_ri = torch.stack([mod.real, mod.imag]).numpy().astype("<f4")

x.numpy().astype("<f4").tofile(out / "torch_stft_in.f32")
ri.tofile(out / "torch_stft_spec.f32")
mod_ri.tofile(out / "torch_istft_spec.f32")
y.numpy().astype("<f4").tofile(out / "torch_istft_out.f32")
print("frames", spec.shape[-1], "torch", torch.__version__)

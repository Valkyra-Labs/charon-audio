#!/usr/bin/env python3
"""Export the music branch of TIGER-DnR to ONNX with the STFT outside.

TIGER-DnR (JusperLee/TIGER, ICLR 2025; weights JusperLee/TIGER-DnR on
Hugging Face, Apache-2.0) is three separate TIGER networks: one per
output (dialog, effects, music), each predicting three sources and used
for one of them. Removing music needs only the music network and only
its music source (index 0), a third of the full model's work.

The exported graph takes the complex STFT of one mono 12 s window as
real and imaginary planes and returns the masked STFT of the music
source; the host computes `torch.stft` and `torch.istft` (n_fft 2048,
hop 512, periodic Hann, centre with reflect padding, not normalized).

Usage:
    export_tiger.py --repo TIGER --weights hf/ --out tiger_music.onnx

Prints parity of (host STFT + ONNX + host iSTFT) against the PyTorch
module on the same window before writing anything else.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SR = 44100
WINDOW_SECONDS = 12.0


class MusicSpec(nn.Module):
    """TIGER.forward from the subband split to the masked spectrum of
    source `k`, on real/imaginary planes (no complex tensors)."""

    def __init__(self, tiger, k=0):
        super().__init__()
        self.t = tiger
        self.k = k

    def forward(self, spec_ri):  # [B, 2, F, T]
        t = self.t
        b = spec_ri.shape[0]
        re, im = spec_ri[:, 0], spec_ri[:, 1]
        feats = []
        idx = 0
        bands = []
        for i, bw in enumerate(t.band_width):
            sub = spec_ri[:, :, idx : idx + bw].contiguous()
            feats.append(t.BN[i](sub.reshape(b, bw * 2, -1)))
            bands.append((idx, bw))
            idx += bw
        feat = torch.stack(feats, 1)
        sep = t.separator(feat.view(b, t.nband, t.feature_dim, -1))
        sep = sep.view(b, t.nband, t.feature_dim, -1)
        out_re, out_im = [], []
        for i, (start, bw) in enumerate(bands):
            o = t.mask[i](sep[:, i]).view(b, 2, 2, t.num_output, bw, -1)
            m = o[:, 0] * torch.sigmoid(o[:, 1])
            mr, mi = m[:, 0], m[:, 1]
            mr = mr - (mr.sum(1, keepdim=True) - 1) / t.num_output
            mi = mi - mi.sum(1, keepdim=True) / t.num_output
            mr, mi = mr[:, self.k], mi[:, self.k]
            sr, si = re[:, start : start + bw], im[:, start : start + bw]
            out_re.append(sr * mr - si * mi)
            out_im.append(sr * mi + si * mr)
        return torch.stack([torch.cat(out_re, 1), torch.cat(out_im, 1)], 1)


class PReLU(nn.Module):
    """nn.PReLU written out: CoreML's MLProgram rejects the exported PReLU
    whose `alpha` has one element for many channels. Same values."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, x):
        w = self.weight
        if w.numel() > 1:
            w = w.view(1, -1, *([1] * (x.dim() - 2)))
        return torch.where(x >= 0, x, w * x)


def replace_prelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.PReLU):
            setattr(module, name, PReLU(child.weight))
        else:
            replace_prelu(child)


def stft(x, n_fft, hop):
    return torch.stft(x, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
                      return_complex=True)


def istft(spec, n_fft, hop, length):
    return torch.istft(spec, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
                       length=length)


def load(repo, weights):
    # Import look2hear.models.tiger_dnr without running the package
    # __init__ files, which pull in training-only dependencies (librosa,
    # torch_complex, lightning). Empty package modules with the right
    # __path__ let the relative imports resolve file by file.
    import importlib
    import types

    root = repo / "look2hear"
    for pkg, path in (("look2hear", root), ("look2hear.layers", root / "layers"),
                      ("look2hear.models", root / "models")):
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(path)]
        sys.modules[pkg] = mod
    TIGERDNR = importlib.import_module("look2hear.models.tiger_dnr").TIGERDNR
    from safetensors.torch import load_file  # noqa: E402

    cfg = json.loads((weights / "config.json").read_text())
    model = TIGERDNR(**cfg)
    state = load_file(str(weights / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="clone of JusperLee/TIGER")
    ap.add_argument("--weights", type=Path, required=True, help="dir with config.json, model.safetensors")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS,
                    help="fixed window length of the export (the authors infer with 12 s)")
    ap.add_argument("--batch", type=int, default=1,
                    help="fixed batch size (channels or windows processed in one run)")
    ap.add_argument("--dynamo", action="store_true")
    ap.add_argument("--coreml-prelu", action="store_true",
                    help="write PReLU out explicitly so CoreML's MLProgram accepts it")
    ap.add_argument("--probe-wav", type=Path, action="append", default=[],
                    help="audio file; 12 s from its middle (mono mix) is added to the parity probes")
    args = ap.parse_args()

    torch.manual_seed(0)
    # PyTorch's CPU path dispatches TIGER's grouped 1-D convolutions to
    # NNPACK one group at a time (135k calls for 0.5 s of audio, 35 s of
    # work); without NNPACK a 12 s window takes about 14 s instead of many
    # minutes. This only affects the reference run, not the export.
    torch.backends.nnpack.set_flags(False)
    model, cfg = load(args.repo, args.weights)
    n_fft, hop = cfg["win"], cfg["stride"]
    music = model.music
    # Reference output before any rewrite, for the wrapper parity below.
    wrapper = MusicSpec(music).eval()
    window = int(SR * args.window_seconds)

    # Parity of the wrapper against the original module, on noise and on
    # a structured signal.
    t = torch.arange(window) / SR
    probes = {
        "noise": 0.1 * torch.randn(1, window),
        "tones": (0.3 * torch.sin(2 * np.pi * 220 * t) * (t % 1 < 0.5)
                  + 0.05 * torch.randn(window))[None],
    }
    for path in args.probe_wav:
        import soundfile as sf

        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if rate != SR:
            sys.exit(f"{path}: {rate} Hz, need {SR}")
        mono = audio.mean(1)
        start = max(0, len(mono) // 2 - window // 2)
        clip = torch.from_numpy(mono[start:start + window].copy())
        if clip.numel() < window:
            clip = torch.nn.functional.pad(clip, (0, window - clip.numel()))
        probes[path.name] = clip[None]
    refs = {}
    with torch.no_grad():
        for name, x in probes.items():
            ref = music(x)[:, 0]  # [1, T], source 0 = music
            refs[name] = ref
            spec = stft(x, n_fft, hop)
            ri = torch.stack([spec.real, spec.imag], 1)
            out = wrapper(ri)
            est = istft(torch.complex(out[:, 0], out[:, 1]), n_fft, hop, window)
            diff = (est - ref).abs().max().item()
            print(f"wrapper vs module [{name}]: max abs diff {diff:.3e}, "
                  f"ref rms {ref.pow(2).mean().sqrt().item():.3e}")
            if diff > 1e-4:
                sys.exit(f"wrapper parity failed on {name}")

    if args.coreml_prelu:
        replace_prelu(music)
        with torch.no_grad():
            for name, x in probes.items():
                spec = stft(x, n_fft, hop)
                ri = torch.stack([spec.real, spec.imag], 1)
                est = istft(torch.complex(*wrapper(ri).unbind(1)), n_fft, hop, window)
                print(f"explicit PReLU vs module [{name}]: max abs diff "
                      f"{(est - refs[name]).abs().max().item():.3e}")

    frames = window // hop + 1
    dummy = torch.zeros(args.batch, 2, n_fft // 2 + 1, frames)
    kwargs = dict(input_names=["spec"], output_names=["music_spec"], opset_version=args.opset)
    if args.dynamo:
        torch.onnx.export(wrapper, (dummy,), str(args.out), dynamo=True, **kwargs)
    else:
        torch.onnx.export(wrapper, (dummy,), str(args.out), dynamo=False, **kwargs)

    import onnx
    import onnxruntime as ort

    # One self-contained file: fold external weight data back in.
    graph = onnx.load(str(args.out), load_external_data=True)
    data_file = args.out.with_name(args.out.name + ".data")
    onnx.save_model(graph, str(args.out), save_as_external_data=False)
    if data_file.exists():
        data_file.unlink()
    onnx.checker.check_model(str(args.out))
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for name, x in probes.items():
            spec = stft(x, n_fft, hop)
            ri = torch.stack([spec.real, spec.imag], 1).repeat(args.batch, 1, 1, 1)
            want = wrapper(ri).numpy()
            got = sess.run(None, {"spec": ri.numpy()})[0][:1]
            want = want[:1]
            est = istft(torch.complex(torch.from_numpy(got[:, 0]), torch.from_numpy(got[:, 1])),
                        n_fft, hop, window)
            ref = music(x)[:, 0]
            rel = ((est - ref).pow(2).sum() / ref.pow(2).sum().clamp_min(1e-20)).item()
            print(f"onnx vs torch [{name}]: spec max diff {np.abs(got - want).max():.3e}, "
                  f"waveform max diff {(est - ref).abs().max().item():.3e}, "
                  f"agreement {-10 * np.log10(max(rel, 1e-20)):.1f} dB, "
                  f"ref rms {ref.pow(2).mean().sqrt().item():.3e}")
    print(json.dumps({
        "file": str(args.out),
        "sha256": sha256(args.out),
        "bytes": args.out.stat().st_size,
        "input": ["spec", [args.batch, 2, n_fft // 2 + 1, frames]],
        "output": ["music_spec", [args.batch, 2, n_fft // 2 + 1, frames]],
        "window_samples": window,
        "n_fft": n_fft,
        "hop": hop,
        "sample_rate": SR,
        "weights_sha256": sha256(args.weights / "model.safetensors"),
        "torch": torch.__version__,
        "onnx_opset": args.opset,
    }, indent=2))


if __name__ == "__main__":
    main()

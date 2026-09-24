"""Export HTDemucs to ONNX with the STFT and iSTFT outside the graph.

Graph contract (batch 1, segment 343980 samples = 7.8 s at 44.1 kHz):
  inputs   mix  [1, 2, 343980]        waveform
           spec [1, 4, 2048, 336]     HTDemucs._magnitude(_spec(mix)):
                                      complex-as-channels, order L.re, L.im,
                                      R.re, R.im; 2048 bins (Nyquist dropped);
                                      336 frames
  outputs  time [1, 4, 2, 343980]     time-branch waveform, denormalized
           spec [1, 4, 4, 2048, 336]  spectral-branch output per source,
                                      denormalized, same layout as input spec
  stems = time + _ispec(spec)   (done by the host)

The network body is demucs 4.1.0 HTDemucs.forward unchanged; the export
patches from demucs-onnx (segment Fraction, pos-embedding randrange,
MultiheadAttention primitives) are reused. A parity check against the
unpatched PyTorch model runs before writing.

Usage: python export_htdemucs.py out/htdemucs_split.onnx [--no-check]
"""
import math, sys, types, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from demucs.pretrained import get_model
from demucs.spec import spectro, ispectro
from demucs_onnx.export.segment import coerce_segment_to_float
from demucs_onnx.export.pos_embed import disable_random_pos_shift
from demucs_onnx.export.mha import onnx_friendly_mha_forward

N_SAMPLES, N_FFT, HOP = 343980, 4096, 1024


def spec_cac(mix: torch.Tensor, model) -> torch.Tensor:
    """HTDemucs._magnitude(HTDemucs._spec(mix)) as a real (B, C*2, F, T) tensor."""
    z = model._spec(mix)
    B, C, Fr, T = z.shape
    return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(B, C * 2, Fr, T)


def ispec_cac(spec: torch.Tensor, model, length: int) -> torch.Tensor:
    """Inverse of spec_cac for a (B, S, C*2, F, T) tensor -> (B, S, C, length)."""
    B, S, C2, Fr, T = spec.shape
    z = spec.view(B, S, C2 // 2, 2, Fr, T).permute(0, 1, 2, 4, 5, 3).contiguous()
    return model._ispec(torch.view_as_complex(z), length)


class SplitHTDemucs(nn.Module):
    """HTDemucs.forward with the transforms moved to the caller."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.captured = None

    def forward(self, mix, spec):
        m = self.model
        B, C2, Fr, T = spec.shape
        # _spec returns what the caller computed; _magnitude reshapes to CAC;
        # _mask is the identity for cac models; _ispec captures the spectral
        # output and contributes nothing to the waveform sum.
        m._spec = types.MethodType(lambda self_, x: spec.view(B, C2 // 2, 2, Fr, T), m)
        m._magnitude = types.MethodType(lambda self_, z: z.reshape(B, C2, Fr, T), m)
        m._mask = types.MethodType(lambda self_, z, x: x, m)

        def _ispec_capture(self_, zout, length=None, scale=0):
            self.captured = zout
            S = zout.shape[1]
            return torch.zeros(B, S, C2 // 2, mix.shape[-1], dtype=mix.dtype)

        m._ispec = types.MethodType(_ispec_capture, m)
        time = m(mix)
        return time, self.captured


def main():
    out = Path(sys.argv[1])
    check = "--no-check" not in sys.argv
    bag = get_model("htdemucs")
    model = bag.models[0]
    model.eval()
    assert list(model.sources) == ["drums", "bass", "other", "vocals"]
    assert model.cac and model.nfft == N_FFT and model.hop_length == HOP

    torch.manual_seed(0)
    mix = torch.randn(1, 2, N_SAMPLES) * 0.1
    with torch.no_grad():
        reference = model(mix)  # unpatched PyTorch, full pipeline
        spec_in = spec_cac(mix, model)
    print("spec input", tuple(spec_in.shape))

    coerce_segment_to_float(model)
    disable_random_pos_shift(model)
    for mod in model.modules():
        if isinstance(mod, nn.MultiheadAttention):
            mod.forward = types.MethodType(onnx_friendly_mha_forward, mod)
    split = SplitHTDemucs(model).eval()

    with torch.no_grad():
        time_out, spec_out = split(mix, spec_in)
        print("outputs", tuple(time_out.shape), tuple(spec_out.shape))
        # Host-side reconstruction against the unpatched model.
        original = get_model("htdemucs").models[0].eval()
        recon = time_out + ispec_cac(spec_out, original, N_SAMPLES)
        diff = (recon - reference).abs().max().item()
        print(f"patched split model vs PyTorch: max |diff| = {diff:.3e}")
        if check:
            assert diff < 1e-3, diff

    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            split, (mix, spec_in), str(out),
            opset_version=17,
            input_names=["mix", "spec"], output_names=["time", "spec_out"],
            do_constant_folding=True, export_params=True, dynamo=False,
        )
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(out)))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    print("onnx inputs", [(i.name, i.shape) for i in sess.get_inputs()])
    print("onnx outputs", [(o.name, o.shape) for o in sess.get_outputs()])
    t, s = sess.run(None, {"mix": mix.numpy(), "spec": spec_in.numpy()})
    with torch.no_grad():
        recon = torch.from_numpy(t) + ispec_cac(torch.from_numpy(s), original, N_SAMPLES)
    diff = (recon - reference).abs().max().item()
    print(f"ONNX (onnxruntime CPU) + host iSTFT vs PyTorch: max |diff| = {diff:.3e}")
    if check:
        assert diff < 1e-3, diff
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({out.stat().st_size} bytes) sha256 {sha}")


if __name__ == "__main__":
    main()

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

The network body is demucs 4.1.0 HTDemucs.forward unchanged; three tracer
patches (segment Fraction, pos-embedding randrange, MultiheadAttention
primitives) follow demucs-onnx 0.3.4 (MIT). A parity check against the
unpatched PyTorch model runs before writing.

Environment: demucs==4.1.0, torch (version printed), onnx, onnxruntime.
Usage: python export_htdemucs.py out/htdemucs_split.onnx [--no-check]
"""
import math, sys, types, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from fractions import Fraction
from demucs.pretrained import get_model
import demucs.transformer as tr

N_SAMPLES, N_FFT, HOP = 343980, 4096, 1024


# The three tracer patches below follow demucs-onnx 0.3.4 (MIT, StemSplit):
# demucs_onnx/export/{segment,pos_embed,mha}.py. They change nothing
# numerically at inference; they remove Python that torch.onnx.export
# cannot trace.

def coerce_segment_to_float(model):
    """model.segment is a Fraction(39, 5); the exporter needs a float."""
    if isinstance(getattr(model, "segment", None), Fraction):
        model.segment = float(model.segment)


def disable_random_pos_shift(model):
    """CrossTransformerEncoder._get_pos_embedding calls random.randrange
    (a no-op at eval since sin_random_shift == 0); hardcode shift = 0."""
    for m in model.modules():
        if hasattr(m, "sin_random_shift"):
            m.sin_random_shift = 0

    def _get_pos_embedding(self_, T, B, C, device):
        if self_.emb == "sin":
            return tr.create_sin_embedding(T, C, shift=0, device=device, max_period=self_.max_period)
        if self_.emb == "cape":
            return tr.create_sin_embedding_cape(
                T, C, B, device=device, max_period=self_.max_period,
                mean_normalize=self_.cape_mean_normalize, augment=False,
                max_global_shift=0.0, max_local_shift=0.0, max_scale=1.0)
        if self_.emb == "scaled":
            return self_.position_embeddings(torch.arange(T, device=device))[:, None]
        raise RuntimeError(f"unknown emb {self_.emb!r}")

    for m in model.modules():
        if isinstance(m, tr.CrossTransformerEncoder):
            m._get_pos_embedding = types.MethodType(_get_pos_embedding, m)


def onnx_friendly_mha_forward(self_, query, key, value, key_padding_mask=None,
                              need_weights=True, attn_mask=None,
                              average_attn_weights=True, is_causal=False):
    """nn.MultiheadAttention.forward from Linear/bmm/softmax only; the fused
    aten::_native_multi_head_attention kernel has no ONNX symbolic."""
    if self_.batch_first:
        query, key, value = (t.transpose(0, 1) for t in (query, key, value))
    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    num_heads = self_.num_heads
    head_dim = embed_dim // num_heads
    if self_._qkv_same_embed_dim:
        w, b = self_.in_proj_weight, self_.in_proj_bias
        w_q, w_k, w_v = w.chunk(3, dim=0)
        b_q, b_k, b_v = b.chunk(3, dim=0) if b is not None else (None, None, None)
        q, k, v = F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)
    else:
        bias = self_.in_proj_bias
        q = F.linear(query, self_.q_proj_weight, bias[:embed_dim] if bias is not None else None)
        k = F.linear(key, self_.k_proj_weight, bias[embed_dim:2 * embed_dim] if bias is not None else None)
        v = F.linear(value, self_.v_proj_weight, bias[2 * embed_dim:] if bias is not None else None)
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1) * head_dim ** -0.5
    k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    attn = torch.bmm(q, k.transpose(1, 2))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = F.softmax(attn, dim=-1)
    out = torch.bmm(attn, v).transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    out = self_.out_proj(out)
    if self_.batch_first:
        out = out.transpose(0, 1)
    if not need_weights:
        return out, None
    attn = attn.view(bsz, num_heads, tgt_len, src_len)
    return out, (attn.mean(dim=1) if average_attn_weights else attn)


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
    import demucs
    print(f"demucs {demucs.__version__}, torch {torch.__version__}")
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

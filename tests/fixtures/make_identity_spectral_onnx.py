"""Build identity_spectral.onnx: the `Spectral` contract (one mono
channel's STFT in as `spec` [1, 2, F, T], one source's STFT out as
`music_spec` [1, 2, F, T]) as an Identity node, for n_fft 64, hop 16 and
1000-sample windows (F = 33, T = 1000 // 16 + 1 = 63). With it the whole
pipeline must return every channel of the input unchanged."""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

F, T = 33, 63
x = helper.make_tensor_value_info("spec", TensorProto.FLOAT, [1, 2, F, T])
y = helper.make_tensor_value_info("music_spec", TensorProto.FLOAT, [1, 2, F, T])
node = helper.make_node("Identity", ["spec"], ["music_spec"])
graph = helper.make_graph([node], "identity_spectral", [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
out = Path(__file__).with_name("identity_spectral.onnx")
onnx.save(model, out)
print(out, out.stat().st_size, "bytes")

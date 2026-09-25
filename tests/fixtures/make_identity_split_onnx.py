"""Generate identity_split.onnx for the DemucsSplit contract:
  mix [1, 2, n], spec [1, 4, f, t]  ->  time [1, 4, 2, n] (mix repeated per
source), spec_out [1, 4, 4, f, t] (zeros). stems = time + ispec(0) = mix.
Requires `onnx`. Run from this directory."""
import onnx
from onnx import TensorProto, helper

mix = helper.make_tensor_value_info("mix", TensorProto.FLOAT, [1, 2, "n"])
spec = helper.make_tensor_value_info("spec", TensorProto.FLOAT, [1, 4, "f", "t"])
time = helper.make_tensor_value_info("time", TensorProto.FLOAT, [1, 4, 2, "n"])
spec_out = helper.make_tensor_value_info("spec_out", TensorProto.FLOAT, [1, 4, 4, "f", "t"])
axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
nodes = [
    helper.make_node("Unsqueeze", ["mix", "axes"], ["m"]),
    helper.make_node("Concat", ["m", "m", "m", "m"], ["time"], axis=1),
    helper.make_node("Mul", ["spec", "zero"], ["s0"]),
    helper.make_node("Unsqueeze", ["s0", "axes"], ["s"]),
    helper.make_node("Concat", ["s", "s", "s", "s"], ["spec_out"], axis=1),
]
graph = helper.make_graph(nodes, "identity_split", [mix, spec], [time, spec_out],
                          initializer=[axes, zero])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "identity_split.onnx")

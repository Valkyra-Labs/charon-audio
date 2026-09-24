"""Generate identity_4stems.onnx: mix [1, 2, n] -> stems [1, 4, 2, n], every
stem a copy of the input. Used to test the pipeline without a real model.
Requires `onnx`. Run from this directory."""
import onnx
from onnx import TensorProto, helper

mix = helper.make_tensor_value_info("mix", TensorProto.FLOAT, [1, 2, "n"])
stems = helper.make_tensor_value_info("stems", TensorProto.FLOAT, [1, 4, 2, "n"])
axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
nodes = [
    helper.make_node("Unsqueeze", ["mix", "axes"], ["x"]),
    helper.make_node("Concat", ["x", "x", "x", "x"], ["stems"], axis=1),
]
graph = helper.make_graph(nodes, "identity_4stems", [mix], [stems], initializer=[axes])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "identity_4stems.onnx")

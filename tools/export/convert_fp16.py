"""Convert the split HTDemucs ONNX to fp16 weights and compute, keeping the
graph inputs and outputs in fp32 (onnxconverter-common float16).

Usage: python convert_fp16.py htdemucs_split.onnx htdemucs_split_fp16.onnx

Prints size and SHA-256. Parity must be measured afterwards; fp16 changes
the numbers (docs/MEASUREMENTS.md)."""
import hashlib, sys
from pathlib import Path
import onnx
from onnxconverter_common import float16

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
model = onnx.load(str(src))
# The exported graph carries explicit Cast(to=float) nodes; the converter
# retypes their outputs without changing `to`, so keep Cast in fp32.
model16 = float16.convert_float_to_float16(model, keep_io_types=True, op_block_list=["Cast"])
onnx.save(model16, str(dst))
print(f"wrote {dst} ({dst.stat().st_size} bytes) sha256 {hashlib.sha256(dst.read_bytes()).hexdigest()}")

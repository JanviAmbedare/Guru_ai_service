from pathlib import Path

import torch
import torch.nn as nn

from facenet_pytorch import (
    InceptionResnetV1
)


# =====================================
# PATHS
# =====================================

BASE_DIR = (
    Path.cwd()
    .resolve()
)

MODEL_DIR = (
    BASE_DIR /
    "models" /
    "exported"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ONNX_PATH = (
    MODEL_DIR /
    "mobilefacenet.onnx"
)


# =====================================
# LOAD MODEL
# =====================================

print(
    "🚀 Loading pretrained FaceNet..."
)

model = (
    InceptionResnetV1(
        pretrained="vggface2"
    )
    .eval()
)

print(
    "✅ Model loaded"
)


# =====================================
# DUMMY INPUT
# =====================================

dummy_input = torch.randn(
    1,
    3,
    160,
    160
)


# =====================================
# EXPORT ONNX
# =====================================

print(
    "🚀 Exporting ONNX..."
)

torch.onnx.export(

    model,

    dummy_input,

    ONNX_PATH,

    export_params=True,

    opset_version=11,

    do_constant_folding=True,

    input_names=["input"],

    output_names=["embedding"],

    dynamic_axes={
        "input": {
            0: "batch_size"
        },

        "embedding": {
            0: "batch_size"
        }
    }
)

print(
    f"✅ ONNX model exported:"
)

print(ONNX_PATH)
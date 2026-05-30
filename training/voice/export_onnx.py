#currently not used, but can be useful for future reference on how to export the model to ONNX
from pathlib import Path

import torch
from speechbrain.inference.speaker import (
    EncoderClassifier
)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PRETRAINED_DIR = (
    BASE_DIR /
    "models" /
    "pretrained" /
    "ecapa_tdnn"
)

EXPORT_DIR = (
    BASE_DIR /
    "models" /
    "exported"
)

PRETRAINED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ONNX_PATH = (
    EXPORT_DIR /
    "ecapa_tdnn.onnx"
)

# ==========================================
# LOAD PRETRAINED MODEL
# ==========================================

print(
    "🚀 Loading ECAPA-TDNN..."
)

classifier = (
    EncoderClassifier
    .from_hparams(
        source=
        "speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(
            PRETRAINED_DIR
        )
    )
)

print(
    "✅ ECAPA downloaded"
)

# ==========================================
# WRAPPER MODULE
# ==========================================

class EcapaWrapper(
    torch.nn.Module
):

    def __init__(
        self,
        classifier
    ):
        super().__init__()

        self.classifier = (
            classifier
        )

    def forward(
        self,
        wav
    ):

        embeddings = (
            self.classifier
            .encode_batch(
                wav
            )
        )

        return embeddings.squeeze(
            1
        )

model = EcapaWrapper(
    classifier
)

model.eval()

# ==========================================
# DUMMY AUDIO
# ==========================================

dummy_audio = torch.randn(
    1,
    16000
)

# ==========================================
# EXPORT
# ==========================================

print(
    "🚀 Exporting ONNX..."
)

torch.onnx.export(
    model,
    dummy_audio,
    str(ONNX_PATH),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=[
        "audio"
    ],
    output_names=[
        "embedding"
    ],
    dynamic_axes={
        "audio": {
            0: "batch_size",
            1: "audio_length"
        },
        "embedding": {
            0: "batch_size"
        }
    }
)

print(
    f"✅ Saved: {ONNX_PATH}"
)

print(
    "🎉 ECAPA-TDNN ONNX export complete"
)
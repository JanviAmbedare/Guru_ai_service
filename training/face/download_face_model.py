from pathlib import Path
import shutil

from insightface.app import FaceAnalysis


# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
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


# =====================================
# DOWNLOAD SMALL MODEL PACK
# =====================================

MODEL_PACK = "buffalo_sc"

print(
    f"🚀 Downloading {MODEL_PACK}..."
)

app = FaceAnalysis(
    name=MODEL_PACK
)

app.prepare(
    ctx_id=0
)

print(
    "✅ Download complete"
)


# =====================================
# FIND RECOGNITION MODEL
# =====================================

INSIGHTFACE_HOME = (
    Path.home()
    / ".insightface"
    / "models"
    / MODEL_PACK
)

onnx_files = list(
    INSIGHTFACE_HOME.glob("*.onnx")
)

print("\n📦 Downloaded ONNX files:")

for file in onnx_files:

    size_mb = (
        file.stat().st_size
        / 1024
        / 1024
    )

    print(
        f"{file.name}"
        f" -> "
        f"{size_mb:.2f} MB"
    )


# =====================================
# COPY RECOGNITION MODEL
# =====================================

recognition_model = None

for file in onnx_files:

    filename = file.name.lower()

    if (
        "mbf" in filename
        or
        "r50" in filename
        or
        "recognition" in filename
    ):
        recognition_model = file
        break

if recognition_model is None:

    raise Exception(
        "Recognition model not found."
    )

destination = (
    MODEL_DIR /
    recognition_model.name
)

shutil.copy2(
    recognition_model,
    destination
)

print(
    "\n✅ Model copied to:"
)

print(destination)

print(
    f"\n📦 Final Size:"
    f" {destination.stat().st_size / 1024 / 1024:.2f} MB"
)
from pathlib import Path
import sys
import os


# =====================================
# ADD PROJECT ROOT TO PYTHON PATH
# =====================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.append(
    str(BASE_DIR)
)

import cv2
import tempfile
import requests
import numpy as np
import onnxruntime as ort

from dotenv import load_dotenv
from retinaface import RetinaFace
from sqlalchemy import text

from utils.database import (
    SessionLocal
)
from utils.file_utils import (
    FileUtils
)
# =====================================
# LOAD ENV
# =====================================

load_dotenv()

# =====================================
# PATHS
# =====================================


MODEL_PATH = (
    BASE_DIR /
    "models" /
    "exported" /
    "w600k_mbf.onnx"
)

EMBEDDING_DIR = (
    BASE_DIR /
    "storage" /
    "embeddings" /
    "faces"
)

EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================
# LOAD ONNX MODEL
# =====================================

print(
    "🚀 Loading ONNX model..."
)

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CPUExecutionProvider"
    ]
)

input_name = (
    session.get_inputs()[0].name
)

print(
    "✅ ONNX model loaded"
)


# =====================================
# PREPROCESS FACE
# =====================================

def preprocess_face(face):

    face = cv2.resize(
        face,
        (112,112)
    )

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face = (face - 127.5) / 127.5

    face = np.transpose(
        face,
        (2, 0, 1)
    )

    face = np.expand_dims(
        face,
        axis=0
    )

    return face


# =====================================
# GENERATE EMBEDDING
# =====================================

def generate_embedding(face):

    processed = preprocess_face(
        face
    )

    embedding = session.run(
        None,
        {
            input_name: processed
        }
    )[0]

    embedding = embedding.flatten()

    norm = np.linalg.norm(
        embedding
    )

    embedding = (
        embedding / norm
    )

    return embedding


# =====================================
# DOWNLOAD IMAGE
# =====================================

def download_image(url):

    response = requests.get(
        url,
        timeout=20
    )

    if response.status_code != 200:

        raise Exception(
            f"Failed to download: {url}"
        )

    suffix = ".jpg"

    temp_file = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )
    )

    temp_file.write(
        response.content
    )

    temp_file.close()

    return temp_file.name


# =====================================
# FETCH FACE RECORDS
# =====================================

def fetch_face_records():

    query = text(
        """
        SELECT
            id,
            user_id,
            cloudinary_url
        FROM media_files
        WHERE media_category='faces'
        AND is_active=1
        """
    )

    db = SessionLocal()

    try:

        results = db.execute(
            query
        ).fetchall()

        return [
            dict(row._mapping)
            for row in results
        ]

    finally:

        db.close()

# =====================================
# PROCESS ALL USERS
# =====================================

def process_faces():

    print(
        "🚀 Fetching face records..."
    )

    records = fetch_face_records()

    print(
        f"✅ Records found: "
        f"{len(records)}"
    )

    user_embeddings = {}

    for record in records:

        temp_path = None

        try:

            user_id = (
                record["user_id"]
            )

            image_url = (
                record["cloudinary_url"]
            )

            print(
                f"\n🚀 Processing "
                f"user {user_id}"
            )

            # ======================
            # DOWNLOAD IMAGE
            # ======================

            temp_path = download_image(
                image_url
            )

            image = cv2.imread(
                temp_path
            )

            if image is None:

                print(
                    "❌ Failed to load image"
                )

                continue

            # ======================
            # DETECT FACE
            # ======================

            faces = RetinaFace.detect_faces(
                temp_path
            )

            if not faces:

                print(
                    "❌ No face detected"
                )

                continue

            first_face = list(
                faces.values()
            )[0]

            x1, y1, x2, y2 = (
                first_face["facial_area"]
            )

            face_crop = image[
                y1:y2,
                x1:x2
            ]

            # ======================
            # GENERATE EMBEDDING
            # ======================

            embedding = (
                generate_embedding(
                    face_crop
                )
            )

            if user_id not in (
                user_embeddings
            ):

                (
                    user_embeddings[user_id]
                ) = []

            (
                user_embeddings[user_id]
                .append(embedding)
            )

            print(
                "✅ Embedding generated"
            )

        except Exception as e:

            print(
                f"❌ Error: {e}"
            )

        finally:

            FileUtils.delete_file(
                    temp_path
                )

    # =================================
    # SAVE FINAL USER EMBEDDINGS
    # =================================

    print(
        "\n🚀 Saving embeddings..."
    )

    for user_id, embeddings in (
        user_embeddings.items()
    ):

        avg_embedding = np.mean(
            embeddings,
            axis=0
        )

        output_path = (
            EMBEDDING_DIR /
            f"{user_id}.npy"
        )

        np.save(
            output_path,
            avg_embedding
        )

        print(
            f"✅ Saved: "
            f"{output_path}"
        )

    print(
        "\n✅ Face enrollment completed"
    )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    process_faces()
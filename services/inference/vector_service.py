import os
import numpy as np

from sqlalchemy import text

from utils.database import (
    SessionLocal
)


class VectorService:

    FACE_DIR = (
        "storage/embeddings/faces"
    )

    VOICE_DIR = (
        "storage/embeddings/voices"
    )

    @staticmethod
    def initialize_storage():

        os.makedirs(
            VectorService.FACE_DIR,
            exist_ok=True
        )

        os.makedirs(
            VectorService.VOICE_DIR,
            exist_ok=True
        )

    # =====================================
    # FACE EMBEDDINGS
    # =====================================

    @staticmethod
    def save_face_embedding(
        user_id,
        embedding,
        sample_count,
        average_quality
    ):

        VectorService.initialize_storage()

        path = (
            f"{VectorService.FACE_DIR}/"
            f"{user_id}.npy"
        )

        np.save(path, embedding)

        db = SessionLocal()

        try:

            db.execute(
                text(
                    """
                    DELETE FROM biometric_profiles
                    WHERE user_id=:user_id
                    AND type='face'
                    """
                ),
                {
                    "user_id": user_id
                }
            )

            db.execute(
                text(
                    """
                    INSERT INTO biometric_profiles
                    (
                        user_id,
                        type,
                        label,
                        file_path,
                        embedding_path,
                        sample_number,
                        is_trained,
                        model_version,
                        quality_score
                    )
                    VALUES
                    (
                        :user_id,
                        'face',
                        :label,
                        :file_path,
                        :embedding_path,
                        :sample_number,
                        1,
                        'mobilefacenet_v1',
                        :average_quality
                    )
                    """
                ),
                {
                    "user_id": user_id,
                    "label": f"user_{user_id}",
                    "file_path":
                        f"guru-storage/faces/raw/{user_id}",
                    "embedding_path": path,
                    "sample_number": sample_count
                }
            )

            db.commit()

        finally:

            db.close()

        return path

    @staticmethod
    def load_face_embedding(user_id):

        path = (
            f"{VectorService.FACE_DIR}/"
            f"{user_id}.npy"
        )

        if not os.path.exists(path):

            return None

        return np.load(path)

    # =====================================
    # VOICE EMBEDDINGS
    # =====================================

    @staticmethod
    def save_voice_embedding(
        user_id,
        embedding,
        sample_count,
        average_quality
    ):

        VectorService.initialize_storage()

        path = (
            f"{VectorService.VOICE_DIR}/"
            f"{user_id}.npy"
        )

        np.save(path, embedding)

        db = SessionLocal()

        try:

            db.execute(
                text(
                    """
                    DELETE FROM biometric_profiles
                    WHERE user_id=:user_id
                    AND type='voice'
                    """
                ),
                {
                    "user_id": user_id
                }
            )

            db.execute(
                text(
                    """
                    INSERT INTO biometric_profiles
                    (
                        user_id,
                        type,
                        label,
                        file_path,
                        embedding_path,
                        sample_number,
                        is_trained,
                        model_version,
                        quality_score
                    )
                    VALUES
                    (
                        :user_id,
                        'voice',
                        :label,
                        :file_path,
                        :embedding_path,
                        :sample_number,
                        1,
                        'ecapa_tdnn_v1',
                        :average_quality
                    )
                    """
                ),
                {
                    "user_id": user_id,
                    "label": f"user_{user_id}",
                    "file_path":
                        f"guru-storage/voices/raw/{user_id}",
                    "embedding_path": path,
                    "sample_number": sample_count,
                    "average_quality": average_quality
                }
            )

            db.commit()

        finally:

            db.close()

        return path

    @staticmethod
    def load_voice_embedding(user_id):

        path = (
            f"{VectorService.VOICE_DIR}/"
            f"{user_id}.npy"
        )

        if not os.path.exists(path):

            return None

        return np.load(path)
import os
import numpy as np

from sqlalchemy import text

from utils.cloudinary_service import CloudinaryService
from utils.database import (
    SessionLocal
)
from pathlib import Path
from services.inference.model_registry_service import (
    ModelRegistryService
)
from services.inference.version_service import (
    VersionService)

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

class VectorService:

    FACE_DIR = (
        BASE_DIR /
        "storage" /
        "embeddings" /
        "faces"
    )

    VOICE_DIR = (
        BASE_DIR /
        "storage" /
        "embeddings" /
        "voices"
    )
    print("VECTOR SERVICE VERSION 19-JUNE")
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

        upload_result = (
            CloudinaryService
            .upload_embedding(
                str(path),
                "faces",
                user_id
            )
        )

        embedding_url = (
            upload_result["url"]
        )

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
            print("FACE INSERT START")
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
                        embedding_url,
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
                        :embedding_url,
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
                    "file_path": "cloudinary",
                    "embedding_path": path,
                    "embedding_url": embedding_url,
                    "sample_number": sample_count,
                    "average_quality": average_quality
                }
            )
            print("FACE INSERT SUCCESS")
            ModelRegistryService.register_model(
                user_id=user_id,
                model_name=f"user_{user_id}_face",
                model_type="face",
                version = VersionService.next_version(
                        user_id,
                        "face"
                    ),
                storage_path=embedding_url
            )
            db.commit()

        except Exception as e:

            print(
                f"VECTOR SERVICE ERROR: {e}"
            )

            raise
        
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

        upload_result = (
            CloudinaryService
            .upload_embedding(
                str(path),
                "voices",
                user_id
            )
        )

        embedding_url = (
            upload_result["url"]
        )

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

            print("VOICE INSERT START")

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
                        embedding_url,
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
                        :embedding_url,
                        :sample_number,
                        1,
                        'mfcc_delta_v1',
                        :average_quality
                    )
                    """
                ),
                {
                    "user_id": user_id,
                    "label": f"user_{user_id}",
                    "file_path": "cloudinary",
                    "embedding_path": path,
                    "embedding_url": embedding_url,
                    "sample_number": sample_count,
                    "average_quality": average_quality
                }
            )
            print("VOICE INSERT SUCCESS")
            ModelRegistryService.register_model(
                    model_name=f"user_{user_id}_voice",
                    user_id=user_id,
                    model_type="voice",
                    version = VersionService.next_version(
                            user_id,
                            "voice"
                        ),
                    storage_path=embedding_url
            )
            db.commit()

        except Exception as e:

            print(
                f"VECTOR SERVICE ERROR: {e}"
            )

            raise
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
import numpy as np

from services.inference.model_manager import (
    ModelManager
)

from services.inference.preprocessing_service import (
    PreprocessingService
)


class FaceEmbeddingService:

    @staticmethod
    def generate_embedding(
        user_id,
        image_path
    ):

        model = (
            ModelManager
            .get_face_model()
        )

        face = (
            PreprocessingService
            .extract_face(
                image_path
            )
        )

        face = (
            PreprocessingService
            .preprocess_face(
                face
            )
        )

        input_name = (
            model.get_inputs()[0].name
        )

        embedding = model.run(
            None,
            {
                input_name: face
            }
        )[0]

        embedding = (
            embedding.flatten()
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm > 0:

            embedding = (
                embedding / norm
            )

        return embedding
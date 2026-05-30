import numpy as np

from services.inference.model_manager import (
    ModelManager
)

from services.inference.preprocessing_service import (
    PreprocessingService
)

from services.inference.vector_service import (
    VectorService
)


class FaceEmbeddingService:

    @staticmethod
    def generate_embedding(
    user_id,
    image_path,
    media_file_id=None
    ):

        try:

            model = (
                ModelManager
                .get_face_model()
            )

            image = (
                PreprocessingService
                .preprocess_face(
                    image_path
                )
            )

            input_name = (
                model.get_inputs()[0].name
            )

            embedding = model.run(
                None,
                {
                    input_name: image
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

        except Exception as e:

            raise Exception(
                f"Face embedding failed: {e}"
            )
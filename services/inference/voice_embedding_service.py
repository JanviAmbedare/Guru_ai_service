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


class VoiceEmbeddingService:

    @staticmethod
    def generate_embedding(
    user_id,
    audio_path,
    media_file_id=None
    ):

        try:

            model = (
                ModelManager
                .get_voice_model()
            )

            audio = (
                PreprocessingService
                .preprocess_voice(
                    audio_path
                )
            )

            input_name = (
                model.get_inputs()[0].name
            )

            embedding = model.run(
                None,
                {
                    input_name: audio
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
                f"Voice embedding failed: {e}"
            )
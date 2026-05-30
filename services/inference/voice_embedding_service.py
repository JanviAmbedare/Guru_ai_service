import numpy as np
import torch

from speechbrain.inference.speaker import (
EncoderClassifier
)

from services.inference.preprocessing_service import (
PreprocessingService
)

class VoiceEmbeddingService:

    _model = None

    @classmethod
    def get_model(cls):

        if cls._model is None:

            print(
                "🚀 Loading ECAPA-TDNN..."
            )

            cls._model = (
                EncoderClassifier
                .from_hparams(
                    source=
                    "speechbrain/spkrec-ecapa-voxceleb",
                    savedir=
                    "models/pretrained/ecapa_tdnn"
                )
            )

            print(
                "✅ ECAPA-TDNN loaded"
            )

        return cls._model

    @classmethod
    def generate_embedding(
        cls,
        audio_path
    ):

        model = cls.get_model()

        audio = (
            PreprocessingService
            .preprocess_voice(
                audio_path
            )
        )

        wav_tensor = (
            torch.tensor(
                audio,
                dtype=torch.float32
            )
        )

        embedding = (
            model
            .encode_batch(
                wav_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )

        embedding = (
            embedding
            .flatten()
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm > 0:

            embedding = (
                embedding / norm
            )

        return embedding
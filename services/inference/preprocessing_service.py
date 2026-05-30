import cv2
import librosa
import numpy as np


class PreprocessingService:

    FACE_SIZE = (112, 112)

    @staticmethod
    def preprocess_face(
        face
    ):

        face = cv2.resize(
            face,
            (112, 112)
        )

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face = face.astype(
            np.float32
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

    @staticmethod
    def preprocess_voice(audio_path):

        audio, sample_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        if len(audio) == 0:

            raise Exception(
                "Empty audio file"
            )

        audio = audio.astype(
            np.float32
        )

        audio = np.expand_dims(
            audio,
            axis=0
        )

        return audio
import cv2
import librosa
import numpy as np


class PreprocessingService:

    FACE_SIZE = (112, 112)

    @staticmethod
    def preprocess_face(image_path):

        image = cv2.imread(image_path)

        if image is None:

            raise Exception(
                f"Unable to load image: "
                f"{image_path}"
            )

        image = cv2.resize(
            image,
            PreprocessingService.FACE_SIZE
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        )

        image = image / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        return image

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
import cv2
import librosa
import numpy as np


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

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
    
    @staticmethod
    def extract_face(image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise Exception(
                f"Cannot load image: {image_path}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80,80)
        )

        if len(faces) == 0:

            return None

        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        face = image[
            y:y+h,
            x:x+w
        ]

        return face
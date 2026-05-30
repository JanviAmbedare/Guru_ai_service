import cv2
import librosa
import numpy as np


class QualityService:

    # ==========================
    # FACE QUALITY
    # ==========================

    @staticmethod
    def calculate_face_quality(
        image_path
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:
            return 0.0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Blur detection

        blur_score = (
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        blur_score = min(
            blur_score / 500,
            1.0
        )

        # Brightness

        brightness = (
            np.mean(gray) / 255
        )

        score = (
            blur_score * 0.6 +
            brightness * 0.4
        )

        return round(
            float(score),
            3
        )

    # ==========================
    # VOICE QUALITY
    # ==========================

    @staticmethod
    def calculate_voice_quality(
        audio_path
    ):

        audio, sr = librosa.load(
            audio_path,
            sr=16000
        )

        if len(audio) == 0:
            return 0.0

        duration = (
            len(audio) / sr
        )

        duration_score = min(
            duration / 5,
            1.0
        )

        energy = (
            np.mean(
                audio ** 2
            )
        )

        energy_score = min(
            energy * 20,
            1.0
        )

        score = (
            duration_score * 0.5 +
            energy_score * 0.5
        )

        return round(
            float(score),
            3
        )
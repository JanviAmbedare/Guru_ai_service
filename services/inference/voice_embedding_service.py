import numpy as np
import torch

import librosa



class VoiceEmbeddingService:

    _model = None

    @staticmethod
    def generate_embedding(audio_path):
        print(
        f"WAV file: "
        f"{audio_path}"
    )
        audio, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )
        print(
            f"Audio duration: "
            f"{len(audio)/sr:.2f} sec"
        )
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )

        mfcc_std = np.std(
            mfcc,
            axis=1
        )

        delta = librosa.feature.delta(
            mfcc
        )

        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        delta_mean = np.mean(
            delta,
            axis=1
        )

        delta2_mean = np.mean(
            delta2,
            axis=1
        )

        embedding = np.concatenate(
            [
                mfcc_mean,
                mfcc_std,
                delta_mean,
                delta2_mean
            ]
        )

        embedding = embedding.astype(
            np.float32
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm > 0:
            embedding = embedding / norm
        print(
                f"Voice embedding shape: "
                f"{embedding.shape}"
            )
        return embedding
from pathlib import Path
import onnxruntime as ort


class ModelManager:

    _face_model = None

    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    FACE_MODEL_PATH = (
        BASE_DIR /
        "models" /
        "exported" /
        "w600k_mbf.onnx"
    )

    @classmethod
    def get_face_model(cls):

        if cls._face_model is None:

            cls._face_model = (
                ort.InferenceSession(
                    str(cls.FACE_MODEL_PATH),
                    providers=[
                        "CPUExecutionProvider"
                    ]
                )
            )

            print(
        "✅ MobileFaceNet ArcFace loaded"
        )

        return cls._face_model
import os
import onnxruntime as ort

class ModelManager:


    _face_model = None

    FACE_MODEL_PATH = (
        "models/exported/mobilefacenet.onnx"
    )

    @classmethod
    def get_face_model(cls):

        if cls._face_model is None:

            if not os.path.exists(
                cls.FACE_MODEL_PATH
            ):

                raise FileNotFoundError(
                    f"Face model not found: "
                    f"{cls.FACE_MODEL_PATH}"
                )

            cls._face_model = (
                ort.InferenceSession(
                    cls.FACE_MODEL_PATH,
                    providers=[
                        "CPUExecutionProvider"
                    ]
                )
            )

            print(
                "✅ Face model loaded"
            )

        return cls._face_model

    @classmethod
    def clear_cache(cls):

        cls._face_model = None

        print(
            "🗑️ Model cache cleared"
        )
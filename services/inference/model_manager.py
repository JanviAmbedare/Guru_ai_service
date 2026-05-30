import os
import onnxruntime as ort


class ModelManager:

    _face_model = None
    _voice_model = None

    FACE_MODEL_PATH = (
        "models/exported/mobilefacenet.onnx"
    )

    VOICE_MODEL_PATH = (
        "models/exported/ecapa_tdnn.onnx"
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
    def get_voice_model(cls):

        if cls._voice_model is None:

            if not os.path.exists(
                cls.VOICE_MODEL_PATH
            ):

                raise FileNotFoundError(
                    f"Voice model not found: "
                    f"{cls.VOICE_MODEL_PATH}"
                )

            cls._voice_model = (
                ort.InferenceSession(
                    cls.VOICE_MODEL_PATH,
                    providers=[
                        "CPUExecutionProvider"
                    ]
                )
            )

            print(
                "✅ Voice model loaded"
            )

        return cls._voice_model
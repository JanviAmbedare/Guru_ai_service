import os
import joblib


class IntentService:

    MODEL_PATH = (
        "models/exported/"
        "intent_pipeline.pkl"
    )

    _intent_pipeline = None

    @classmethod
    def load_model(cls):

        if cls._intent_pipeline is None:

            if not os.path.exists(
                cls.MODEL_PATH
            ):

                raise FileNotFoundError(
                    f"Intent model not found: "
                    f"{cls.MODEL_PATH}"
                )

            cls._intent_pipeline = (
                joblib.load(
                    cls.MODEL_PATH
                )
            )

            print(
                "✅ Intent model loaded"
            )

        return cls._intent_pipeline

    @classmethod
    def detect_intent(
        cls,
        text: str
    ):

        try:

            model = cls.load_model()

            prediction = (
                model.predict([text])[0]
            )

            return prediction

        except Exception as e:

            print(
                f"Intent detection error: {e}"
            )

            return "fallback"
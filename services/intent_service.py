import joblib

intent_pipeline = joblib.load(
    "models/intent_pipeline.pkl"
)

class IntentService:

    @staticmethod
    def detect_intent(text: str):

        try:
            prediction = (
                intent_pipeline
                .predict([text])[0]
            )

            return prediction

        except:
            return "fallback"
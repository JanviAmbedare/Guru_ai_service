class SentimentService:

    @staticmethod
    def detect_sentiment(text: str):

        text = text.lower()

        positive_words = [
            "happy",
            "good",
            "great",
            "awesome",
            "excellent"
        ]

        negative_words = [
            "sad",
            "bad",
            "angry",
            "depressed",
            "stress"
        ]

        if any(w in text for w in positive_words):
            return "positive"

        if any(w in text for w in negative_words):
            return "negative"

        return "neutral"
class SentimentService:

    POSITIVE_WORDS = [
        "happy",
        "good",
        "great",
        "awesome",
        "excellent",
        "love",
        "nice"
    ]

    NEGATIVE_WORDS = [
        "sad",
        "bad",
        "angry",
        "depressed",
        "stress",
        "pain",
        "upset"
    ]

    @classmethod
    def detect_sentiment(
        cls,
        text: str
    ):

        text = text.lower()

        positive_score = sum(
            word in text
            for word in (
                cls.POSITIVE_WORDS
            )
        )

        negative_score = sum(
            word in text
            for word in (
                cls.NEGATIVE_WORDS
            )
        )

        if positive_score > negative_score:

            return "positive"

        elif negative_score > positive_score:

            return "negative"

        return "neutral"
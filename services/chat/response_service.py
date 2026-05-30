class ResponseService:

    @staticmethod
    def generate_response(
        text,
        intent,
        sentiment,
        memories
    ):

        memory_context = ""

        for memory in memories[-5:]:

            memory_context += (
                f"{memory}\n"
            )

        response = (
            f"Intent detected: {intent}. "
            f"Sentiment: {sentiment}. "
        )

        if intent == "greeting":

            response += (
                "Hello! How can I help you?"
            )

        elif intent == "emergency":

            response += (
                "Emergency detected. "
                "Alerting system."
            )

        elif sentiment == "negative":

            response += (
                "I understand you may "
                "be feeling stressed."
            )

        else:

            response += (
                "I understand your message."
            )

        return response
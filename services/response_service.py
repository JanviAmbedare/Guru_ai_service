class ResponseService:

    @staticmethod
    def generate_response(
        text,
        intent,
        sentiment,
        memories
    ):

        memory_context = ""

        for m in memories:

            memory_context += f"{m}\n"

        response = (
            f"I understand your message. "
            f"Detected intent: {intent}. "
            f"Emotion: {sentiment}."
        )

        return response
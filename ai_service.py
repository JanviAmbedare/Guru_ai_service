from services.intent_service import (
    IntentService
)

from services.sentiment_service import (
    SentimentService
)

from services.memory_service import (
    MemoryService
)

from services.response_service import (
    ResponseService
)

class GuruAIService:

    @staticmethod
    def process_message(
        user_id,
        text
    ):

        # 🎯 INTENT
        intent = (
            IntentService
            .detect_intent(text)
        )

        # 😊 SENTIMENT
        sentiment = (
            SentimentService
            .detect_sentiment(text)
        )

        # 🧠 MEMORY
        memories = (
            MemoryService
            .get_memories(user_id)
        )

        # 💬 RESPONSE
        response = (
            ResponseService
            .generate_response(
                text,
                intent,
                sentiment,
                memories
            )
        )

        # 💾 SAVE MEMORY
        MemoryService.save_memory(
            user_id,
            text
        )

        return {
            "intent": intent,
            "sentiment": sentiment,
            "response": response
        }
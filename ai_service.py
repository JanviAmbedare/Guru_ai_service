from services.chat.intent_service import (
    IntentService
)

from services.chat.sentiment_service import (
    SentimentService
)

from services.chat.memory_service import (
    MemoryService
)

from services.chat.response_service import (
    ResponseService
)
from services.inference.recognition_service import (
    RecognitionService
)

from services.inference.face_embedding_service import (
    FaceEmbeddingService
)

from services.inference.voice_embedding_service import (
    VoiceEmbeddingService
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
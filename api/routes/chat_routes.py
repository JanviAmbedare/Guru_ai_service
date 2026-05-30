from fastapi import (
    APIRouter
)

from pydantic import (
    BaseModel
)

from sqlalchemy import text

from utils.database import (
    SessionLocal
)

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


router = APIRouter()


# =====================================
# REQUEST MODEL
# =====================================

class ChatRequest(BaseModel):

    user_id: int
    message: str


# =====================================
# CHAT API
# =====================================

@router.post("/chat")

def process_chat(
    request: ChatRequest
):

    db = SessionLocal()

    try:

        # ==========================
        # INTENT
        # ==========================

        intent = (
            IntentService
            .detect_intent(
                request.message
            )
        )

        # ==========================
        # SENTIMENT
        # ==========================

        sentiment = (
            SentimentService
            .detect_sentiment(
                request.message
            )
        )

        # ==========================
        # MEMORY
        # ==========================

        memories = (
            MemoryService
            .get_memories(
                request.user_id
            )
        )

        # ==========================
        # RESPONSE
        # ==========================

        response = (
            ResponseService
            .generate_response(
                request.message,
                intent,
                sentiment,
                memories
            )
        )

        # ==========================
        # SAVE MEMORY
        # ==========================

        MemoryService.save_memory(
            request.user_id,
            request.message
        )

        # ==========================
        # SAVE CONVERSATION
        # ==========================

        conversation_query = text(
            """
            INSERT INTO conversations_v2
            (
                user_id,
                user_message,
                ai_response,
                intent,
                sentiment
            )
            VALUES
            (
                :user_id,
                :user_message,
                :ai_response,
                :intent,
                :sentiment
            )
            """
        )

        db.execute(
            conversation_query,
            {
                "user_id": (
                    request.user_id
                ),

                "user_message": (
                    request.message
                ),

                "ai_response": response,

                "intent": intent,

                "sentiment": sentiment
            }
        )

        db.commit()

        return {
            "status": "success",
            "intent": intent,
            "sentiment": sentiment,
            "response": response
        }

    except Exception as e:

        return {
            "status": "failed",
            "error": str(e)
        }

    finally:

        db.close()
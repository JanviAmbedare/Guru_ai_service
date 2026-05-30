from fastapi import APIRouter
from pydantic import BaseModel

from sqlalchemy import text

from utils.database import SessionLocal
from ai_service import GuruAIService

router = APIRouter()


class ChatRequest(BaseModel):

    user_id: int
    message: str


@router.post("/chat")
def process_chat(
    request: ChatRequest
):

    db = SessionLocal()

    try:

        result = (
            GuruAIService
            .process_message(
                request.user_id,
                request.message
            )
        )

        session_id = (
            f"user_{request.user_id}"
        )

        # User message

        db.execute(
            text(
                """
                INSERT INTO conversations_v2
                (
                    user_id,
                    session_id,
                    role,
                    message,
                    intent,
                    emotion
                )
                VALUES
                (
                    :user_id,
                    :session_id,
                    'user',
                    :message,
                    :intent,
                    :emotion
                )
                """
            ),
            {
                "user_id": request.user_id,
                "session_id": session_id,
                "message": request.message,
                "intent": result["intent"],
                "emotion": result["emotion"]
            }
        )

        # Assistant response

        db.execute(
            text(
                """
                INSERT INTO conversations_v2
                (
                    user_id,
                    session_id,
                    role,
                    message,
                    intent,
                    emotion
                )
                VALUES
                (
                    :user_id,
                    :session_id,
                    'assistant',
                    :message,
                    :intent,
                    :emotion
                )
                """
            ),
            {
                "user_id": request.user_id,
                "session_id": session_id,
                "message": result["response"],
                "intent": result["intent"],
                "emotion": result["emotion"]
            }
        )

        db.commit()

        return {
            "status": "success",
            **result
        }

    except Exception as e:

        db.rollback()

        return {
            "status": "failed",
            "error": str(e)
        }

    finally:

        db.close()
from fastapi import APIRouter
from sqlalchemy import text

from fastapi import HTTPException
from utils.database import SessionLocal

router = APIRouter(
    prefix="/api/models",
    tags=["Global Models"]
)


@router.get("/global/intent")
def get_intent_model():

    db = SessionLocal()

    try:

        result = db.execute(
            text(
                """
                SELECT *
                FROM global_model_registry
                WHERE
                    model_type = 'intent'
                    AND active = 1
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
        )

        model = result.mappings().first()

        if not model:

            raise HTTPException(
                status_code=404,
                detail="Intent model not found"
            )

        return model

    finally:

        db.close()
        
@router.get("/global/emotion")
def get_emotion_model():

    db = SessionLocal()

    try:

        result = db.execute(
            text(
                """
                SELECT *
                FROM global_model_registry
                WHERE
                    model_type = 'emotion'
                    AND active = 1
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
        )

        model = result.mappings().first()

        if not model:

            raise HTTPException(
                status_code=404,
                detail="Emotion model not found"
            )

        return model

    finally:

        db.close()
from fastapi import APIRouter

from sqlalchemy import text

from utils.database import (
    SessionLocal
)

router = APIRouter()


@router.get(
    "/training/status/{user_id}"
)
def get_training_status(
    user_id: int
):

    db = SessionLocal()

    try:

        query = text(
            """
            SELECT
                type,
                status,
                completed_at
            FROM model_training_queue
            WHERE user_id=:user_id
            ORDER BY created_at DESC
            """
        )

        rows = db.execute(
            query,
            {
                "user_id": user_id
            }
        ).fetchall()

        return [
            dict(
                row._mapping
            )
            for row in rows
        ]

    finally:

        db.close()
from fastapi import APIRouter
from sqlalchemy import text

from utils.database import SessionLocal

router = APIRouter()


@router.get("/models/latest/{user_id}")
def latest_models(user_id: int):

    db = SessionLocal()

    try:

        rows = db.execute(
            text(
                """
                SELECT *
                FROM model_registry
                WHERE user_id=:user_id
                AND active=1
                """
            ),
            {
                "user_id": user_id
            }
        ).fetchall()

        return [
            dict(row._mapping)
            for row in rows
        ]

    finally:
        db.close()
from fastapi import APIRouter
from sqlalchemy import text

from utils.database import SessionLocal
from fastapi import APIRouter

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

@router.get(
    "/models/global"
)
def get_global_models():
    db = SessionLocal()
    
    try:
        query = """
        SELECT *
        FROM global_model_registry
        WHERE active = 1
        ORDER BY created_at DESC
        """

        return db.execute(
            query
        ).fetchall()
        
    finally:
        db.close()
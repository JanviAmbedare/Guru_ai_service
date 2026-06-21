from fastapi import APIRouter

from sqlalchemy import text

from utils.database import (
    SessionLocal
)
from services.queue.queue_monitor_service import (
    QueueMonitorService
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
        
@router.post("/training/retrain/{user_id}")
def retrain_user(user_id:int):

    db = SessionLocal()

    try:

        face_count = db.execute(
            text("""
                SELECT COUNT(*)
                FROM media_files
                WHERE user_id=:user_id
                AND media_category='faces'
                AND is_active=1
            """),
            {"user_id": user_id}
        ).scalar()

        voice_count = db.execute(
            text("""
                SELECT COUNT(*)
                FROM media_files
                WHERE user_id=:user_id
                AND media_category='voices'
                AND is_active=1
            """),
            {"user_id": user_id}
        ).scalar()

        QueueMonitorService.force_retrain(
            user_id,
            "face",
            face_count
        )

        QueueMonitorService.force_retrain(
            user_id,
            "voice",
            voice_count
        )

        return {
            "status":"success",
            "message":"Retraining queued"
        }

    finally:
        db.close()
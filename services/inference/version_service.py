from sqlalchemy import text

from utils.database import SessionLocal


class VersionService:

    @staticmethod
    def next_version(
        user_id,
        model_type
    ):

        db = SessionLocal()

        try:

            row = db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM model_registry
                    WHERE user_id=:user_id
                    AND model_type=:model_type
                    """
                ),
                {
                    "user_id": user_id,
                    "model_type": model_type
                }
            ).fetchone()

            count = row[0]

            return f"v{count + 1}"

        finally:
            db.close()
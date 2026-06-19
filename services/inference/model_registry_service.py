from sqlalchemy import text

from utils.database import SessionLocal


class ModelRegistryService:

    @staticmethod
    def register_model(
        user_id,
        model_name,
        model_type,
        version,
        storage_path,
        checksum=None
    ):

        db = SessionLocal()

        try:

            db.execute(
                text(
                    """
                    UPDATE model_registry
                    SET active=0
                    WHERE user_id=:user_id
                    AND model_type=:model_type
                    """
                ),
                {
                    "user_id": user_id,
                    "model_type": model_type
                }
            )

            db.execute(
                text(
                    """
                    INSERT INTO model_registry
                    (
                        user_id,
                        model_name,
                        model_type,
                        version,
                        storage_path,
                        checksum,
                        active
                    )
                    VALUES
                    (
                        :user_id,
                        :model_name,
                        :model_type,
                        :version,
                        :storage_path,
                        :checksum,
                        1
                    )
                    """
                ),
                {
                    "user_id": user_id,
                    "model_name": model_name,
                    "model_type": model_type,
                    "version": version,
                    "storage_path": storage_path,
                    "checksum": checksum
                }
            )

            db.commit()

        finally:
            db.close()
import time

from sqlalchemy import text

from utils.database import (
    SessionLocal
)

from services.queue.training_service import (
    TrainingService
)


class QueueMonitorService:

    WORKER_SLEEP = 5

    @staticmethod
    def start_worker():

        print(
            "🚀 Queue worker started"
        )

        while True:

            db = SessionLocal()

            try:

                query = text(
                    """
                    SELECT *
                    FROM model_training_queue
                    WHERE status IN
                    (
                        'pending',
                        'uploading',
                        'queued'
                    )
                    ORDER BY created_at ASC
                    LIMIT 1
                    """
                )

                result = db.execute(
                    query
                ).fetchone()
                print(f"Worker result: {result}")
                if result:

                    job = dict(
                        result._mapping
                    )

                    db.execute(
                        text(
                            """
                            UPDATE model_training_queue
                            SET
                                status='processing',
                                started_at=NOW()
                            WHERE id=:job_id
                            """
                        ),
                        {
                            "job_id": job["id"]
                        }
                    )

                    db.commit()

                    print(
                        f"⚙ Processing job: "
                        f"{job['id']}"
                    )

                    TrainingService.process_job(
                        db,
                        job
                    )

            except Exception as e:

                print(
                    f"❌ Queue worker error: "
                    f"{e}"
                )

            finally:

                db.close()

            time.sleep(
                QueueMonitorService
                .WORKER_SLEEP
            )
            
    @staticmethod
    def force_retrain(
        user_id,
        job_type,
        total_files
    ):
        conn = SessionLocal().connection().connection
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE media_files
            SET is_used_for_training=0
            WHERE user_id=%s
            AND media_category='faces'
            """,
            (user_id,)
        )

        cursor.execute(
            """
            UPDATE media_files
            SET is_used_for_training=0
            WHERE user_id=%s
            AND media_category='voices'
            """,
            (user_id,)
        )

        conn.commit()
        cursor.execute(
            """
            UPDATE model_training_queue
            SET status='cancelled'
            WHERE user_id=%s
            AND type=%s
            AND status IN
            (
                'pending',
                'uploading',
                'processing',
                'training'
            )
            """,
            (
                user_id,
                job_type
            )
        )

        cursor.execute(
            """
            INSERT INTO model_training_queue
            (
                user_id,
                type,
                status,
                total_files,
                processed_files,
                progress_percentage
            )
            VALUES
            (
                %s,
                %s,
                'pending',
                %s,
                0,
                0
            )
            """,
            (
                user_id,
                job_type,
                total_files
            )
        )

        conn.commit()

        cursor.close()
        conn.close()

        return {
            "status": "queued"
        }
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
                    WHERE status='pending'
                    ORDER BY created_at ASC
                    LIMIT 1
                    """
                )

                result = db.execute(
                    query
                ).fetchone()

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
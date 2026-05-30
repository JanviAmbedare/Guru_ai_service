from sqlalchemy import text
from utils.database import (
    SessionLocal
)
import numpy as np
from services.inference.face_embedding_service import (
    FaceEmbeddingService
)

from services.inference.voice_embedding_service import (
    VoiceEmbeddingService
)

from services.inference.vector_service import (
    VectorService
)
from services.inference.quality_service import (
    QualityService
)
from utils.cloudinary_service import (
    CloudinaryService
)

class TrainingService:

    @staticmethod
    def process_job(
        db,
        job
    ):

        try:

            job_id = job["id"]

            user_id = job["user_id"]

            job_type = job["type"]

            media_query = text(
                """
                SELECT *
                FROM media_files
                WHERE user_id=:user_id
                AND media_category=:category
                AND is_active=1
                AND is_used_for_training=0
                ORDER BY created_at ASC
                """
            )

            media_results = db.execute(
            media_query,
            {
                "user_id": user_id,
                "category": (
                    "faces"
                    if job_type == "face"
                    else "voices"
                )
            }
        ).fetchall()

            if not media_results:
                raise Exception(
                    "Media files not found"
                )
            quality_scores = []
            embeddings = []

            for row in media_results:

                media = dict(
                    row._mapping
                )

                media_path = (
                    CloudinaryService.download_file(
                        media["cloudinary_url"]
                    )
                )

                if job_type == "face":

                    embedding = (
                        FaceEmbeddingService.generate_embedding(
                            user_id,
                            media_path
                        )
                    )
                    quality_scores.append(

                        QualityService
                        .calculate_face_quality(
                            media_path
                        )
                    )

                else:

                    embedding = (
                        VoiceEmbeddingService.generate_embedding(
                            media_path
                        )
                    )
                    quality_scores.append(

                        QualityService
                        .calculate_voice_quality(
                            media_path
                        )
                    )

                embeddings.append(
                        embedding
                    )
            if not embeddings:
                raise Exception(
                    "No embeddings generated"
                )
            average_quality = (
                    sum(quality_scores)
                    /
                    len(quality_scores)
                )   
            final_embedding = np.mean(
                embeddings,
                axis=0
            )

            norm = np.linalg.norm(
                final_embedding
            )

            if norm == 0:
                raise Exception(
                    "Invalid embedding norm"
                )

            final_embedding = (
                final_embedding / norm
            )
            for row in media_results:

                media = dict(
                    row._mapping
                )

                media_id = media["id"]

                update_query = text(
                    """
                    UPDATE media_files
                    SET is_used_for_training=1
                    WHERE id=:media_id
                    """
                )

                db.execute(
                    update_query,
                    {
                        "media_id": media_id
                    }
                )
            db.commit()

            if job_type == "face":

                VectorService.save_face_embedding(
                        user_id,
                        final_embedding,
                        len(media_results),
                        average_quality
                    )

            else:

                VectorService.save_voice_embedding(
                        user_id,
                        final_embedding,
                        len(media_results),
                        average_quality
                    )
            
            complete_query = text(
                """
                UPDATE model_training_queue
                SET
                    status='completed',
                    completed_at=NOW()
                WHERE id=:job_id
                """
            )

            db.execute(
                complete_query,
                {
                    "job_id": job_id
                }
            )

            db.commit()


            print(
                f"✅ Job completed: "
                f"{job_id}"
            )

        except Exception as e:

            error_query = text(
                """
                UPDATE model_training_queue
                SET status='failed',
                    error_message=:error
                WHERE id=:job_id
                """
            )

            db.execute(
                error_query,
                {
                    "job_id": job["id"],
                    "error": str(e)
                }
            )

            db.commit()

            print(
                f"❌ Training failed: "
                f"{e}"
            )
import os
import tempfile

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form
)

from sqlalchemy import text

from utils.database import (
    SessionLocal
)

from utils.cloudinary_service import (
    CloudinaryService
)

from utils.file_utils import (
    FileUtils
)

from services.inference.vector_service import (
    VectorService
)

from services.inference.recognition_service import (
    RecognitionService
)

from services.inference.voice_embedding_service import (
    VoiceEmbeddingService
)


router = APIRouter()


# =====================================
# VOICE REGISTRATION
# =====================================

@router.post("/voice/register")

async def register_voice(

    user_id: int = Form(...),

    file: UploadFile = File(...)
):

    db = SessionLocal()

    temp_path = None

    try:

        suffix = os.path.splitext(
            file.filename
        )[1]

        temp_file = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )
        )

        contents = await file.read()

        temp_file.write(contents)

        temp_file.close()

        temp_path = temp_file.name

        upload_result = (
            CloudinaryService
            .upload_file(
                temp_path,
                "guru/voices"
            )
        )

        media_query = text(
            """
            INSERT INTO media_files
            (
                user_id,
                media_category,
                media_role,
                file_name,
                local_path,
                cloudinary_url,
                public_id,
                file_size,
                mime_type,
                upload_source,
                is_processed,
                is_active
            )
            VALUES
            (
                :user_id,
                'voices',
                'registration',
                :file_name,
                :local_path,
                :cloudinary_url,
                :public_id,
                :file_size,
                :mime_type,
                'guru-ai',
                0,
                1
            )
            """
        )

        result = db.execute(
            media_query,
            {
                "user_id": user_id,
                "file_name": file.filename,
                "local_path": temp_path,
                "cloudinary_url": (
                    upload_result["url"]
                ),
                "public_id": (
                    upload_result["public_id"]
                ),
                "file_size": len(contents),
                "mime_type": file.content_type
            }
        )

        db.commit()

        media_file_id = (
            result.lastrowid
        )

        queue_query = text(
            """
            INSERT INTO model_training_queue
            (
                user_id,
                media_file_id,
                type,
                status
            )
            VALUES
            (
                :user_id,
                :media_file_id,
                'voice',
                'pending'
            )
            """
        )

        db.execute(
            queue_query,
            {
                "user_id": user_id,
                "media_file_id": (
                    media_file_id
                )
            }
        )

        db.commit()

        return {
            "status": "success",
            "message": (
                "Voice uploaded "
                "and queued"
            ),
            "media_file_id": (
                media_file_id
            )
        }

    except Exception as e:

        return {
            "status": "failed",
            "error": str(e)
        }

    finally:

        db.close()

        if temp_path:

            FileUtils.delete_file(
                temp_path
            )


# # =====================================
# # VOICE RECOGNITION
# # =====================================

# @router.post("/voice/recognize")

# async def recognize_voice(

#     user_id: int = Form(...),

#     file: UploadFile = File(...)
# ):

#     temp_path = None

#     try:

#         suffix = os.path.splitext(
#             file.filename
#         )[1]

#         temp_file = (
#             tempfile.NamedTemporaryFile(
#                 delete=False,
#                 suffix=suffix
#             )
#         )

#         contents = await file.read()

#         temp_file.write(contents)

#         temp_file.close()

#         temp_path = temp_file.name

#         new_embedding = (
#             VoiceEmbeddingService
#             .generate_embedding(
#                 user_id,
#                 temp_path
#             )
#         )

#         stored_embedding = (
#             VectorService
#             .load_voice_embedding(
#                 user_id
#             )
#         )

#         if stored_embedding is None:

#             return {
#                 "status": "failed",
#                 "message": (
#                     "No registered voice"
#                 )
#             }

#         result = (
#             RecognitionService
#             .is_voice_match(
#                 stored_embedding,
#                 new_embedding
#             )
#         )

#         return {
#             "status": "success",
#             "matched": (
#                 result["matched"]
#             ),
#             "similarity": (
#                 result["similarity"]
#             )
#         }

#     except Exception as e:

#         return {
#             "status": "failed",
#             "error": str(e)
#         }

#     finally:

#         if temp_path:

#             FileUtils.delete_file(
#                 temp_path
#             )
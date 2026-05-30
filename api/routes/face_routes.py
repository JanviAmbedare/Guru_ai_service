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

router = APIRouter()


# =====================================
# FACE REGISTRATION
# =====================================

@router.post("/face/register")
async def register_face(

    user_id: int = Form(...),

    file: UploadFile = File(...)

):

    db = SessionLocal()

    temp_path = None

    try:

        # ==========================
        # SAVE TEMP FILE
        # ==========================

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

        temp_file.write(
            contents
        )

        temp_file.close()

        temp_path = (
            temp_file.name
        )

        # ==========================
        # CLOUDINARY UPLOAD
        # ==========================

        upload_result = (
            CloudinaryService
            .upload_file(
                temp_path,
                "guru/faces"
            )
        )

        # ==========================
        # SAVE MEDIA RECORD
        # ==========================

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
                'faces',
                'registration',
                :file_name,
                NULL,
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

        # ==========================
        # TRAINING QUEUE
        # ==========================

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
                'face',
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
                "Face uploaded and queued"
            ),
            "media_file_id": (
                media_file_id
            ),
            "cloudinary_url": (
                upload_result["url"]
            )
        }

    except Exception as e:

        db.rollback()

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
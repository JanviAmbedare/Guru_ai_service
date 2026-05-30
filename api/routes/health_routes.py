from fastapi import APIRouter

from sqlalchemy import text

from utils.database import (
    SessionLocal
)

from services.inference.model_manager import (
    ModelManager
)


router = APIRouter()


# =====================================
# BASIC HEALTH
# =====================================

@router.get("/")

def health_check():

    return {
        "status": "success",
        "message": (
            "GURU AI running"
        )
    }


# =====================================
# PING
# =====================================

@router.get("/ping")

def ping():

    return {
        "message": "pong"
    }


# =====================================
# SYSTEM STATUS
# =====================================

@router.get("/status")

def system_status():

    db = SessionLocal()

    try:

        # ==========================
        # DATABASE CHECK
        # ==========================

        db.execute(
            text("SELECT 1")
        )

        db_status = "connected"

        # ==========================
        # MODEL CHECK
        # ==========================

        face_model = (
            ModelManager
            .get_face_model()
        )

        voice_model = (
            ModelManager
            .get_voice_model()
        )

        return {
            "status": "success",

            "database": db_status,

            "face_model": (
                "loaded"
                if face_model
                else "not_loaded"
            ),

            "voice_model": (
                "loaded"
                if voice_model
                else "not_loaded"
            ),

            "api": "running"
        }

    except Exception as e:

        return {
            "status": "failed",
            "error": str(e)
        }

    finally:

        db.close()
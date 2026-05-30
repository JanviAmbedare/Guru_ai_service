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

        return {
            "status": "success",

            "database": db_status,

            "api": "running"
        }

    except Exception as e:

        return {
            "status": "failed",
            "error": str(e)
        }

    finally:

        db.close()
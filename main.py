from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from api.routes.health_routes import (
    router as health_router
)

from api.routes.chat_routes import (
    router as chat_router
)

from api.routes.face_routes import (
    router as face_router
)

from api.routes.voice_routes import (
    router as voice_router
)

from services.queue.queue_monitor_service import (
    QueueMonitorService
)
from services.inference.model_manager import (
    ModelManager
)
from api.routes.training_routes import (
    router as training_router
)

app = FastAPI(
    title="GURU AI",
    description=(
        "AI inference engine for "
        "GURU Assistive Companion Robot"
    ),
    version="1.0.0"
)


# ==============================
# CORS
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# ROUTES
# ==============================

app.include_router(
    health_router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"]
)

app.include_router(
    face_router,
    prefix="/api",
    tags=["Face"]
)

app.include_router(
    voice_router,
    prefix="/api",
    tags=["Voice"]
)

app.include_router(
    training_router,
    prefix="/api",
    tags=["Training"]
)

# ==============================
# STARTUP EVENTS
# ==============================
@app.on_event("startup")
async def startup_event():

    print(
        "🚀 GURU AI starting..."
    )

    # =====================
    # LOAD MODELS
    # =====================

    try:

        ModelManager.load_face_model()

        print(
            "✅ Face model loaded"
        )

    except Exception as e:

        print(
            f"❌ Face model error: {e}"
        )

    try:

        ModelManager.load_voice_model()

        print(
            "✅ Voice model loaded"
        )

    except Exception as e:

        print(
            f"❌ Voice model error: {e}"
        )

    # =====================
    # START QUEUE WORKER
    # =====================

    worker_thread = threading.Thread(
        target=QueueMonitorService.start_worker,
        daemon=True
    )

    worker_thread.start()

    print(
        "✅ Queue worker started"
    )

    print(
        "✅ APIs initialized"
    )

# ==============================
# ROOT ENDPOINT
# ==============================

@app.get("/")

def root():

    return {
        "status": "success",
        "message": (
            "GURU AI running"
        ),
        "version": "1.0.0"
    }
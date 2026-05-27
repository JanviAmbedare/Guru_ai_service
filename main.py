from fastapi import FastAPI
from pydantic import BaseModel

from ai_service import GuruAIService

app = FastAPI()

class ChatRequest(BaseModel):

    user_id: int
    message: str

@app.get("/")
def home():

    return {
        "message": "Guru AI Service Running"
    }

@app.post("/chat")
def chat(request: ChatRequest):

    result = (
        GuruAIService.process_message(
            request.user_id,
            request.message
        )
    )

    return result
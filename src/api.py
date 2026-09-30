from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.route_request import process_request


app = FastAPI(
    title="Voice SIP Router API",
    description="BERT-based voice request intent classification and SIP routing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Voice SIP Router API is running"
    }


@app.post("/route")
def route_voice_request(request: VoiceRequest):

    intent, confidence, endpoint = process_request(
        request.text
    )

    return {
        "text": request.text,
        "intent": intent,
        "confidence": confidence,
        "sip_endpoint": endpoint
    }
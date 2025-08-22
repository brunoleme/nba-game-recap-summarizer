from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError, Field, validator
from loguru import logger
import os
from pathlib import Path
import psycopg2
import requests

from text2cypher.api.config import settings
from text2cypher.finetuning.models.t5_model import T5NoteGenerationModel
from text2cypher.finetuning.data.notechat_dataset import NoteChatDataModule
from text2cypher.finetuning.utils.logger import setup_logger
from text2cypher.finetuning.utils.text_utils import clean_conversation

DECODER_HOST = os.getenv("DECODER_HOST", "http://localhost:8001")

# Initialize model variable at module level
model = None

app = FastAPI(
    title="Clinical Notes Generator API",
    description="API for generating clinical notes from doctor-patient conversations",
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Setup logging
setup_logger()

# Initialize model at startup
@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    try:
        global model
        model = T5NoteGenerationModel.load_model_from_checkpoint(
            checkpoint_path=str(settings.model_path),
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Failed to initialize model")

@app.get("/api")
async def root():
    return {
        "name": "Clinical Notes Generator API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)",
            "/api": "This information",
            "/health": "Health check endpoint",
            "/generate_note": "Generate clinical notes from conversations"
        }
    }

class ConversationRequest(BaseModel):
    conversation: str = Field(..., min_length=1, description="The doctor-patient conversation text")
    max_length: int = Field(default=512, ge=1, le=1024, description="Maximum length of generated note")

    @validator('conversation')
    def clean_conversation_input(cls, v):
        logger.info("Validating conversation input")
        try:
            v = v.replace('\n', ' ').replace('\r', ' ')
            v = ' '.join(v.split())
            logger.info("Conversation cleaned in validator")
            return v
        except Exception as e:
            logger.error(f"Error in conversation validator: {str(e)}")
            raise ValueError(f"Invalid conversation format: {str(e)}")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation": "Doctor: How are you feeling today? Patient: I have a headache.",
                "max_length": 512
            }
        }

class NoteResponse(BaseModel):
    clinical_note: str

    class Config:
        schema_extra = {
            "example": {
                "clinical_note": "Patient presents with headache..."
            }
        }

@app.post("/generate_note", response_model=NoteResponse)
async def generate_note(request: ConversationRequest):
    logger.info("Incoming request to /generate_note endpoint")
    logger.debug("Raw request received")
    try:
        logger.info("Received generate_note request")
        logger.debug(f"Original request: {request.dict()}")
        if not request.conversation:
            raise HTTPException(status_code=400, detail="Empty conversation")
        conversation = clean_conversation(NoteChatDataModule.format_conversation(request.conversation))
        logger.info("Generating clinical note...")
        clinical_note = model.generate_note(conversation=conversation, max_length=request.max_length)
        logger.info("Note generation successful")
        return NoteResponse(clinical_note=clinical_note)
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating note: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_note_decoupled", response_model=NoteResponse)
async def generate_note_with_decoupling(request: ConversationRequest):
    logger.info("Received request for distributed generation")
    conversation = f"summarize: {NoteChatDataModule.format_conversation(request.conversation)}"
    conversation = clean_conversation(conversation)
    prefill_data = model.prefill(conversation, max_length=request.max_length)
    decode_response = requests.post(
        f"{DECODER_HOST}/decode",
        json={
            "encoder_hidden_states": prefill_data["encoder_hidden_states"],
            "attention_mask": prefill_data["attention_mask"],
            "max_length": request.max_length,
        },
        timeout=30,
    )
    if decode_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Decoding failed")
    clinical_note = decode_response.json()["generated_note"]
    return NoteResponse(clinical_note=clinical_note)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} request to {request.url}")
    try:
        body = await request.body()
        if body:
            logger.debug(f"Request body: {body.decode()}")
    except Exception as e:
        logger.error(f"Could not log request body: {str(e)}")
    response = await call_next(request)
    return response

@app.post("/enqueue")
async def enqueue(request: ConversationRequest):
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO inference_queue (conversation) VALUES (%s)", (request.conversation,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "queued"}
    except Exception as e:
        logger.exception("DB insert failed")
        raise HTTPException(status_code=500, detail="Failed to queue request")

@app.get("/trigger_batch_inference")
async def trigger_batch():
    try:
        from src.batch_jobs.run_batch_inference import main as batch_main
        batch_main()
        return {"status": "success", "message": "Batch inference triggered."}
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed.")

@app.get("/trigger_batch_inference_decoupled")
async def trigger_batch_decoupled():
    try:
        from src.batch_jobs.run_batch_inference_decoupled import main as batch_main
        batch_main()
        return {"status": "success", "message": "Batch inference triggered."}
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed.")

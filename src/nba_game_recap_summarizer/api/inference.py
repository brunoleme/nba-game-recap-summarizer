from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError, Field, validator
from loguru import logger
import os

from nba_game_recap_summarizer.api.config import settings
from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.utils.logger import setup_logger
from nba_game_recap_summarizer.finetuning.utils.text_utils import clean_game_recap

# Initialize model variable at module level
model = None

app = FastAPI(
    title="NBA Game Recap Summarizer API",
    description="API for generating recap summaries from ESPN NBA recaps",
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
        
        # Try to load from Hugging Face format first
        hf_model_path = "/app/models/hf_model"
        if os.path.exists(hf_model_path) and os.path.exists(os.path.join(hf_model_path, "config.json")):
            logger.info(f"Loading model from Hugging Face format: {hf_model_path}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Load tokenizer and model directly from HF format
            tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
            model_hf = AutoModelForCausalLM.from_pretrained(
                hf_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Create model instance with loaded components
            model = LlamaRecapSummarizationModel(
                model_name="meta-llama/Llama-3.2-1B-Instruct",
                tokenizer=tokenizer,
                model_hf=model_hf
            )
        else:
            # Fallback to checkpoint loading or S3 download
            logger.info("Hugging Face model not found, attempting S3 download...")
            
            # Download model from S3 if it's an S3 path
            if str(settings.model_path).startswith("s3://"):
                import boto3
                import tempfile
                import zipfile
                
                # Parse S3 path
                s3_path = str(settings.model_path)
                bucket_name = s3_path.split("/")[2]
                key = "/".join(s3_path.split("/")[3:])
                
                # Create local directory
                os.makedirs(hf_model_path, exist_ok=True)
                
                # Download from S3
                s3_client = boto3.client('s3')
                logger.info(f"Downloading model from S3: s3://{bucket_name}/{key}")
                
                # Try to download as a directory (multiple files)
                try:
                    paginator = s3_client.get_paginator('list_objects_v2')
                    pages = paginator.paginate(Bucket=bucket_name, Prefix=key)
                    
                    for page in pages:
                        if 'Contents' in page:
                            for obj in page['Contents']:
                                file_key = obj['Key']
                                local_file_path = os.path.join(hf_model_path, file_key.replace(key, '').lstrip('/'))
                                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                                s3_client.download_file(bucket_name, file_key, local_file_path)
                    
                    logger.info("Model downloaded successfully from S3")
                    
                    # Load the downloaded model
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                    
                    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
                    model_hf = AutoModelForCausalLM.from_pretrained(
                        hf_model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    
                    model = LlamaRecapSummarizationModel(
                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                        tokenizer=tokenizer,
                        model_hf=model_hf
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to download model from S3: {str(e)}")
                    raise
            else:
                # Fallback to checkpoint loading
                logger.info(f"Loading model from checkpoint: {settings.model_path}")
                model = LlamaRecapSummarizationModel.load_model_from_checkpoint(
                    checkpoint_path=str(settings.model_path),
                )
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Failed to initialize model")

@app.get("/api")
async def root():
    return {
        "name": "NBA Game Recap Summarizer API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)",
            "/api": "This information",
            "/health": "Health check endpoint",
            "/summarize_recap": "Generate recap summaries from NBA game recaps"
        }
    }

class GameRecapRequest(BaseModel):
    game_recap: str = Field(..., min_length=1, description="The NBA game recap")
    max_length: int = Field(default=2048, ge=1, le=2048, description="Maximum length of generated recap summary")

    @validator('game_recap')
    def clean_game_recap_input(cls, v):
        logger.info("Validating game_recap input")
        try:
            v = v.replace('\n', ' ').replace('\r', ' ')
            v = ' '.join(v.split())
            logger.info("game_recap cleaned in validator")
            return v
        except Exception as e:
            logger.error(f"Error in game_recap validator: {str(e)}")
            raise ValueError(f"Invalid game_recap format: {str(e)}")

    class Config:
        json_schema_extra = {
            "example": {
                "game_recap": "Lakers beats Suns in the overtime with a Bryant game winner.",
                "max_length": 2048
            }
        }

class GameRecapResponse(BaseModel):
    game_recap_summary: str

    class Config:
        schema_extra = {
            "example": {
                "game_recap_summary": "Lakers beats Suns..."
            }
        }

@app.post("/summarize_recap", response_model=GameRecapResponse)
async def summarize_recap(request: GameRecapRequest):
    logger.info("Incoming request to /summarize_recap endpoint")
    logger.debug("Raw request received")
    try:
        logger.info("Received summarize_recap request")
        logger.debug(f"Original request: {request.dict()}")
        if not request.game_recap:
            raise HTTPException(status_code=400, detail="Empty game recap")
        game_recap = clean_game_recap(request.game_recap)
        logger.info("Generating game recap summary...")
        game_recap_summary = model.summarize_recap(game_recap=game_recap, max_length=request.max_length)
        logger.info("Game recap summarization with success")
        return GameRecapResponse(game_recap_summary=game_recap_summary)
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error summarizing game recap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint that verifies model is loaded."""
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded or not ready")
    return {"status": "healthy", "model_loaded": True}

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

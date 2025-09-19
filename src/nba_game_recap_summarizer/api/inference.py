from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError, Field, validator
from loguru import logger
import os

from nba_game_recap_summarizer.api.config import settings
from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.models.phi_model import PhiRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.models.mistral_model import MistralRecapSummarizationModel
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
        
        # Determine model type from environment or config
        model_type = os.getenv("MODEL_TYPE", "llama").lower()
        
        # Check if Hugging Face model was downloaded during build time
        local_model_path = "/app/models/hf_model"
        if os.path.exists(local_model_path):
            logger.info(f"Loading {model_type} model from Hugging Face format: {local_model_path}")
            # Load Hugging Face format model directly
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model_hf = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Create model wrapper
            if model_type == "phi":
                model = PhiRecapSummarizationModel(
                    model_name=local_model_path,
                    tokenizer=tokenizer,
                    model=model_hf
                )
            elif model_type == "mistral":
                model = MistralRecapSummarizationModel(
                    model_name=local_model_path,
                    tokenizer=tokenizer,
                    model=model_hf
                )
            else:  # Default to LLaMA
                model = LlamaRecapSummarizationModel(
                    model_name=local_model_path,
                    tokenizer=tokenizer,
                    model=model_hf
                )
        else:
            # Download Hugging Face model from S3 first
            s3_model_path = str(settings.model_path).replace(".ckpt", "/hf_model")
            logger.info(f"Local model not found, downloading from S3: {s3_model_path}")
            try:
                import boto3
                import shutil
                s3_client = boto3.client('s3')
                
                # Extract bucket and key from S3 path
                s3_path_parts = s3_model_path.replace("s3://", "").split("/", 1)
                bucket_name = s3_path_parts[0]
                object_key = s3_path_parts[1]
                
                # Download model directory
                logger.info(f"Downloading Hugging Face model from s3://{bucket_name}/{object_key}")
                # Create local directory
                os.makedirs(local_model_path, exist_ok=True)
                
                # List and download all files in the S3 directory
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=object_key)
                for obj in response.get('Contents', []):
                    file_key = obj['Key']
                    local_file_path = os.path.join(local_model_path, file_key.replace(object_key, "").lstrip("/"))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    s3_client.download_file(bucket_name, file_key, local_file_path)
                
                logger.success(f"Hugging Face model downloaded successfully to {local_model_path}")
                
                # Now load from local path using Hugging Face format
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                model_hf = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Create model wrapper
                if model_type == "phi":
                    model = PhiRecapSummarizationModel(
                        model_name=local_model_path,
                        tokenizer=tokenizer,
                        model=model_hf
                    )
                elif model_type == "mistral":
                    model = MistralRecapSummarizationModel(
                        model_name=local_model_path,
                        tokenizer=tokenizer,
                        model=model_hf
                    )
                else:  # Default to LLaMA
                    model = LlamaRecapSummarizationModel(
                        model_name=local_model_path,
                        tokenizer=tokenizer,
                        model=model_hf
                    )
                    
            except Exception as e:
                logger.error(f"Failed to download model from S3: {e}")
                logger.info("Falling back to Hugging Face model")
                # Fallback to Hugging Face model
                if model_type == "phi":
                    model = PhiRecapSummarizationModel(
                        model_name="microsoft/Phi-3.5-mini-instruct",
                        use_quantization=True,
                        quantization_type="4bit",
                        peft_method="lora",
                    )
                elif model_type == "mistral":
                    model = MistralRecapSummarizationModel(
                        model_name="mistralai/Mistral-7B-Instruct-v0.3",
                        use_quantization=True,
                        quantization_type="4bit",
                        peft_method="lora",
                    )
                else:  # Default to LLaMA
                    model = LlamaRecapSummarizationModel(
                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                        use_quantization=True,
                        quantization_type="4bit",
                        peft_method="lora",
                    )
        
        logger.info(f"{model_type.upper()} model loaded successfully")
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

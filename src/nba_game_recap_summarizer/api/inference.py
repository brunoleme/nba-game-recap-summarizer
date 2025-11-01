from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError, Field, field_validator, ConfigDict
from loguru import logger
import os
import time

from nba_game_recap_summarizer.api.config import settings
from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.utils.logger import setup_logger
from nba_game_recap_summarizer.finetuning.utils.text_utils import clean_game_recap
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

# Initialize model variable at module level
model = None

app = FastAPI(
    title="NBA Game Recap Summarizer API",
    description="API for generating recap summaries from ESPN NBA recaps",
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Expose Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint", "status_code"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

EXCEPTIONS_COUNT = Counter(
    "http_exceptions_total",
    "Total HTTP exceptions",
    ["method", "endpoint", "exception_type"],
)

MODEL_LOADED_GAUGE = Gauge(
    "model_loaded",
    "Whether the model is loaded (1) or not (0)",
)

SUMMARY_GEN_DURATION = Histogram(
    "summary_generation_duration_seconds",
    "Time spent generating a summary",
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0),
)

SUMMARY_LENGTH_CHARS = Histogram(
    "summary_length_chars",
    "Length of generated summaries in characters",
    buckets=(100, 200, 400, 800, 1200, 2000, 4000, 8000),
)

# Setup logging
setup_logger()

# Initialize model at startup
@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    try:
        global model
        
        loaded_successfully = False

        # Load from local Hugging Face format (prefer aligned, then merged)
        for hf_model_path in ["/app/models/hf_model_merged_aligned", "/app/models/hf_model_merged"]:
            if os.path.exists(hf_model_path) and os.path.exists(os.path.join(hf_model_path, "config.json")):
                logger.info(f"Loading model from Hugging Face format: {hf_model_path}")
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
                loaded_successfully = True
                break

        if not loaded_successfully:
            # Fallback to checkpoint loading or S3 download
            logger.info("Hugging Face model not found, attempting S3 download...")

            if str(settings.model_path).startswith("s3://"):
                import boto3

                # Parse S3 path
                s3_path = str(settings.model_path)
                bucket_name = s3_path.split("/")[2]
                key = "/".join(s3_path.split("/")[3:])

                try:
                    # Download model from S3. Try aligned first, then merged, then hf_model
                    base_key = key.rstrip('/')
                    candidate_subdirs = [
                        "hf_model_merged_aligned",
                        "hf_model_merged",
                    ]

                    # Attempt download for the first existing subdir
                    local_path = "/app/models/hf_model_merged"
                    os.makedirs(local_path, exist_ok=True)

                    s3_client = boto3.client('s3')
                    files_downloaded = 0
                    for subdir in candidate_subdirs:
                        s3_prefix = f"{base_key}/{subdir}"
                        logger.info(f"Trying S3 model path: s3://{bucket_name}/{s3_prefix}")
                        paginator = s3_client.get_paginator('list_objects_v2')
                        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
                        found_any = False
                        for page in pages:
                            if 'Contents' in page:
                                found_any = True
                                for obj in page['Contents']:
                                    file_key = obj['Key']
                                    local_file_path = os.path.join(local_path, file_key.replace(s3_prefix, '').lstrip('/'))
                                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                                    s3_client.download_file(bucket_name, file_key, local_file_path)
                                    files_downloaded += 1
                        if found_any and files_downloaded > 0:
                            logger.info(f"Downloaded model from subdir: {subdir}")
                            break

                    if files_downloaded > 0:
                        logger.info(f"Model downloaded successfully from S3 ({files_downloaded} files)")

                        # Load the downloaded model
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        import torch

                        tokenizer = AutoTokenizer.from_pretrained(local_path)
                        model_hf = AutoModelForCausalLM.from_pretrained(
                            local_path,
                            torch_dtype=torch.float16,
                            device_map="auto"
                        )

                        model = LlamaRecapSummarizationModel(
                            model_name="meta-llama/Llama-3.2-1B-Instruct",
                            tokenizer=tokenizer,
                            model_hf=model_hf
                        )
                        loaded_successfully = True
                    else:
                        logger.warning("No files found in S3 path; skipping S3 model load")
                except Exception as e:
                    logger.error(f"Failed to download model from S3: {str(e)}")
                    raise
            else:
                # Fallback to checkpoint loading
                logger.info(f"Loading model from checkpoint: {settings.model_path}")
                model = LlamaRecapSummarizationModel.load_model_from_checkpoint(
                    checkpoint_path=str(settings.model_path),
                )
                loaded_successfully = True
        
        if loaded_successfully:
            logger.info("Model loaded successfully")
            MODEL_LOADED_GAUGE.set(1)
        else:
            logger.warning("Model not loaded; service will report unhealthy until model is available")
            MODEL_LOADED_GAUGE.set(0)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        MODEL_LOADED_GAUGE.set(0)
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

    @field_validator('game_recap')
    @classmethod
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
        start_gen = time.perf_counter()
        game_recap_summary = model.summarize_recap(game_recap=game_recap, max_length=request.max_length)
        gen_duration = time.perf_counter() - start_gen
        SUMMARY_GEN_DURATION.observe(gen_duration)
        SUMMARY_LENGTH_CHARS.observe(len(game_recap_summary))
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
    start_time = time.perf_counter()
    method = request.method
    endpoint = request.url.path
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        status_code = str(response.status_code)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status_code=status_code).observe(duration)
        return response
    except Exception as e:
        duration = time.perf_counter() - start_time
        EXCEPTIONS_COUNT.labels(method=method, endpoint=endpoint, exception_type=e.__class__.__name__).inc()
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code="500").inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status_code="500").observe(duration)
        raise

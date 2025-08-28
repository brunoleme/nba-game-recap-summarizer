from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    model_path: str = "checkpoints/best_model.ckpt"

settings = Settings(
    model_path=os.getenv("MODEL_PATH", Settings().model_path)
)
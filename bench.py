from models.base import get_model
from settings import get_settings, get_logger
import mlflow


settings = get_settings()
logger = get_logger(__name__)

model, tokenizer = get_model()

# WARN: WIP
raise NotImplementedError("WIP")

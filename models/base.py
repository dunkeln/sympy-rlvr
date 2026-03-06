from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from settings import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

accelerator = Accelerator()
logger.info("Initializing base model on device=%s", accelerator.device)
logger.info("Selected model profile=%s repo=%s", settings.model.value, settings.model_repo_id)

try:
    logger.info("Trying local model cache for %s", settings.model_repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_repo_id, dtype="auto", local_files_only=True
    )
    logger.info("Loaded model from local cache")
except OSError:
    logger.warning(
        "Local cache miss for %s; falling back to Hugging Face download",
        settings.model_repo_id,
    )
    model = AutoModelForCausalLM.from_pretrained(settings.model_repo_id, dtype="auto")

model.config.use_cache = False
logger.info("Base model initialized with use_cache=%s", model.config.use_cache)

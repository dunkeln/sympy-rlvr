from functools import lru_cache

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import get_logger, get_settings


logger = get_logger(__name__)
settings = get_settings()

accelerator = Accelerator()
logger.info("Initializing base model on device=%s", accelerator.device)
logger.info(
    "Selected model profile=%s repo=%s", settings.model.value, settings.model_repo_id
)


@lru_cache(maxsize=1)
def get_model():
    try:
        logger.info("Trying local model/tokenizer cache for %s", settings.model_repo_id)
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_repo_id,
            dtype="bfloat16",
            local_files_only=True,
        ).to(accelerator.device)
        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_repo_id, local_files_only=True
        )
        logger.info("Loaded model/tokenizer from local cache")
    except OSError:
        logger.warning(
            "Local cache miss for %s; falling back to Hugging Face download",
            settings.model_repo_id,
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_repo_id,
            dtype="bfloat16",
        ).to(accelerator.device)
        tokenizer = AutoTokenizer.from_pretrained(settings.model_repo_id)

    model.config.use_cache = False
    logger.info(
        "Loaded model=%s, use_cache=%s", settings.model_repo_id, model.config.use_cache
    )
    return model, tokenizer

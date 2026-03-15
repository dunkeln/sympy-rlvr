from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from settings import get_logger, get_settings
from functools import lru_cache

logger = get_logger(__name__)
settings = get_settings()

accelerator = Accelerator()
logger.info("Initializing partial LoRA model on device=%s", accelerator.device)
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
        "Loaded model=%s,  use_cache=%s", settings.model_repo_id, model.config.use_cache
    )

    lora_partial = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # trains attn layers
            "gate_proj",
            "up_proj",
            "down_proj",  # trains mlp layers
        ],
        layers_to_transform=list(range(20, 27)),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_partial).to(accelerator.device)
    logger.info("Applied partial LoRA adapter to model")
    return model, tokenizer

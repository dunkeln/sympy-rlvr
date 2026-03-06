from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from project_logging import get_logger

logger = get_logger(__name__)

accelerator = Accelerator()
logger.info("Initializing base model on device=%s", accelerator.device)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype="auto")
model.config.use_cache = False
logger.info("Base model initialized with use_cache=%s", model.config.use_cache)

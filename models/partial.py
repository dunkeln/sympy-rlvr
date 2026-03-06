from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from project_logging import get_logger

logger = get_logger(__name__)

accelerator = Accelerator()
logger.info("Initializing partial LoRA model on device=%s", accelerator.device)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype="auto")
model.config.use_cache = False
logger.info("Loaded base model for partial LoRA with use_cache=%s", model.config.use_cache)

qlora_partial = LoraConfig(
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

model = get_peft_model(model, qlora_partial).to(accelerator.device)
logger.info("Applied partial LoRA adapter to model")

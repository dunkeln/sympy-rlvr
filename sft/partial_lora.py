from ..models.partial import model
from accelerate import Accelerator
import torch
from project_logging import get_logger

logger = get_logger(__name__)

accelerator = Accelerator()
logger.info("SFT partial LoRA module initialized on device=%s", accelerator.device)

from pathlib import Path
from models.partial import get_model
from accelerate import Accelerator
from settings import get_logger
from settings import MlflowMode, get_settings
import torch
import torch.nn as nn
from sft.data_prep import load_gsm8k_train, transform_gsm8k_resp
from datasets import Dataset


logger = get_logger(__name__)
settings = get_settings()

_GSM8K_XML_PARQUET_PATH = Path("data/gsm8k-main-train-xml.parquet")

if settings.set_mlflow is MlflowMode.ON:
    import mlflow

    logger.info("mlflow enabled")


accelerator = Accelerator()
logger.info(f"{__file__} initialized on device=%s", accelerator.device)

model, tokenizer = get_model()

try:
    dataset = Dataset.from_parquet(_GSM8K_XML_PARQUET_PATH.as_posix())
except FileNotFoundError:
    dataset = load_gsm8k_train()
    dataset = dataset.map(transform_gsm8k_resp)
    dataset.to_parquet(_GSM8K_XML_PARQUET_PATH.as_posix())
    logger.info("saved XML reformatted text to parquet")

    logger.info(f"{__file__} transformation to SFT required response completed")


def train():
    raise NotImplementedError()

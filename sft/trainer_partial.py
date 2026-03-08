"""Partial-LoRA SFT entrypoint over XML-transformed GSM8K data.

Purpose:
- Load model, acquire transformed dataset, and host training loop scaffolding.

Owns:
- Dataset bootstrap logic (`data/gsm8k-main-train-xml.parquet`).
- Top-level training lifecycle hooks (`step`, `epoch`, `train`).

If you need to change behavior:
- Adjust data loading near module init, then implement `step` and `epoch`.
"""

from pathlib import Path

from models.partial import get_model
from accelerate import Accelerator
from settings import get_logger
from settings import MlflowMode, get_settings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sft.data_prep import load_gsm8k_train, transform_gsm8k_resp
from sft.qwen_tok_dataset import QwenTokDataset
from datasets import Dataset
import click


logger = get_logger(__name__)
settings = get_settings()

_GSM8K_XML_PARQUET_PATH = Path("data/gsm8k-main-train-xml.parquet")

if settings.set_mlflow is MlflowMode.ON:
    import mlflow

    logger.info("mlflow enabled")


accelerator = Accelerator()
logger.info(f"{__file__} initialized on device=%s", accelerator.device)

# INFO: load XML formatted dataset first
try:
    dataset = Dataset.from_parquet(_GSM8K_XML_PARQUET_PATH.as_posix())
except FileNotFoundError:
    dataset = load_gsm8k_train()
    dataset = dataset.map(transform_gsm8k_resp)
    dataset.to_parquet(_GSM8K_XML_PARQUET_PATH.as_posix())
    logger.info("saved XML reformatted text to parquet")

    logger.info(f"{__file__} transformation to SFT required response completed")


def step(row):
    """Execute one training step for a tokenized batch row."""
    raise NotImplementedError()


def epoch(model, tokenizer):
    """Run one full pass over the training dataset."""
    raise NotImplementedError()


@click.command()
@click.option("--alpha", type=click.FLOAT, help="learning rate for backward pass")
@click.option("--epochs", default=1, help="num epochs for SFT training")
def train(epochs: int = 3, alpha: float = 0.001):
    """Run SFT for a fixed number of epochs.

    Args:
        epochs (int): Number of training epochs to execute.
    """
    model, tokenizer = get_model()
    optim = torch.optim.Adam(model.parameters(), lr=alpha)
    loss = nn.CrossEntropyLoss()
    dataloader = DataLoader(QwenTokDataset(dataset, tokenizer))

    for ep in range(1, epochs + 1):
        epoch(model, tokenizer)

        # TODO: validation post-training
    raise NotImplementedError()


if __name__ == "__main__":
    raise NotImplementedError()

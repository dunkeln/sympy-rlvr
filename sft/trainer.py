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
from typing import Any

import click
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from models.partial import get_model
from sft.data_prep import load_gsm8k_train, transform_gsm8k_resp
from sft.qwen_tok_dataset import QwenTokDataset
from settings import get_logger


logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"{__file__} running on device={device}")


# INFO: Load and transform GSM8K
_GSM8K_XML_PARQUET_PATH = Path("data/gsm8k-main-train-xml.parquet")

try:
    dataset = Dataset.from_parquet(_GSM8K_XML_PARQUET_PATH.as_posix())
except FileNotFoundError:
    dataset = load_gsm8k_train()
    dataset = dataset.map(transform_gsm8k_resp)
    dataset.to_parquet(_GSM8K_XML_PARQUET_PATH.as_posix())
    logger.info("saved XML reformatted text to parquet")
    logger.info(f"{__file__} transformation to SFT required response completed")


# INFO: Building blocks for training
def step(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    row: dict[str, torch.Tensor],
) -> float:
    """Execute one optimizer step for a tokenized training batch."""
    model.train()
    optim.zero_grad(set_to_none=True)
    batch = {name: tensor.to(device) for name, tensor in row.items()}

    outputs = model(**batch)
    loss = outputs.loss
    if not torch.isfinite(loss.detach()):
        raise RuntimeError("Encountered non-finite loss during SFT step")

    loss.backward()
    optim.step()

    return float(loss.detach().item())


def epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
    epoch_idx: int,
) -> float:
    """Run one training epoch and return the mean batch loss."""
    total_loss = 0.0
    num_batches = 0

    logger.info("sft_epoch_start epoch=%s", epoch_idx)
    for batch_idx, row in enumerate(dataloader, start=1):
        batch_loss = step(model, optim, row)
        total_loss += batch_loss
        num_batches = batch_idx

        if batch_idx == 1 or batch_idx % 25 == 0:
            logger.info(
                "sft_batch_complete epoch=%s batch=%s loss=%.6f",
                epoch_idx,
                batch_idx,
                batch_loss,
            )

    if num_batches == 0:
        raise ValueError("Training dataloader yielded zero batches")

    avg_loss = total_loss / num_batches
    logger.info(
        "sft_epoch_complete epoch=%s batches=%s avg_loss=%.6f",
        epoch_idx,
        num_batches,
        avg_loss,
    )
    return avg_loss


@click.command()
@click.option(
    "--alpha",
    type=click.FLOAT,
    default=0.001,
    show_default=True,
    help="learning rate for backward pass",
)
@click.option(
    "--epochs", default=1, show_default=True, help="num epochs for SFT training"
)
def train(epochs: int = 1, alpha: float = 0.001) -> dict[str, Any]:
    """Run SFT for a fixed number of epochs."""
    model, tokenizer = get_model()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=alpha)
    tok_dataset = QwenTokDataset(dataset, tokenizer)
    dataloader = DataLoader(
        tok_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=tok_dataset.collate_fn,
    )

    logger.info(
        "sft_train_start epochs=%s alpha=%s dataset_size=%s batch_size=%s",
        epochs,
        alpha,
        len(tok_dataset),
        32,
    )

    final_epoch_loss = 0.0
    for ep in range(1, epochs + 1):
        final_epoch_loss = epoch(model, dataloader, optim, ep)

    summary = {
        "epochs": epochs,
        "learning_rate": alpha,
        "dataset_size": len(tok_dataset),
        "final_epoch_loss": final_epoch_loss,
    }

    logger.info(
        "sft_train_complete epochs=%s final_epoch_loss=%.6f",
        epochs,
        final_epoch_loss,
    )
    return summary


if __name__ == "__main__":
    train()

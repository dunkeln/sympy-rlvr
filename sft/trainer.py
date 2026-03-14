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
from typing import Any, Iterable, Optional

import click
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from models.partial import get_model
from sft.data_prep import load_gsm8k_train, transform_gsm8k_resp
from sft.qwen_tok_dataset import QwenTokDataset
from settings import get_logger, get_settings

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


logger = get_logger(__name__)
settings_conf = get_settings()

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
        if mlflow is not None:
            mlflow.set_tag("error", "Encountered non-finite loss during SFT step")
        raise RuntimeError("Encountered non-finite loss during SFT step")

    loss.backward()
    optim.step()

    return float(loss.detach().item())


def _evaluate_loss(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
) -> Optional[float]:
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for row in dataloader:
            batch_val = {name: tensor.to(device) for name, tensor in row.items()}
            loss = model(**batch_val).loss
            total_loss += float(loss.detach().item())
            batches += 1

    if batches == 0:
        return None
    return total_loss / batches


def epoch(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    optim: torch.optim.Optimizer,
    epoch_idx: int,
    val_dataloader: Optional[Iterable[dict[str, torch.Tensor]]] = None,
) -> tuple[float, Optional[float]]:
    """Run one training epoch and return the mean batch loss."""
    total_loss = 0.0
    num_batches = 0

    logger.info("sft_epoch_start epoch=%s", epoch_idx)
    for batch_idx, row in enumerate(dataloader, start=1):
        batch_loss = step(model, optim, row)
        total_loss += batch_loss
        num_batches = batch_idx

        if batch_idx == 1 or batch_idx % 25 == 0:
            metrics = {
                "epoch": epoch_idx,
                "batch_idx": batch_idx,
                "train/loss": batch_loss,
            }
            if mlflow is not None:
                mlflow.log_metrics(metrics, step=batch_idx)
            else:
                logger.info("sft batch data", metrics)

    if num_batches == 0:
        raise ValueError("Training dataloader yielded zero batches")

    avg_loss = total_loss / num_batches
    if mlflow is not None:
        mlflow.log_metric("epoch/loss", avg_loss, step=epoch_idx)
    else:
        logger.info(
            "sft_epoch_complete epoch=%s batches=%s avg_loss=%.6f",
            epoch_idx,
            num_batches,
            avg_loss,
        )
    val_loss = None
    if val_dataloader is not None:
        val_loss = _evaluate_loss(model, val_dataloader)
        if val_loss is not None:
            logger.info(
                "sft_validation_epoch epoch=%s val_loss=%.6f",
                epoch_idx,
                val_loss,
            )
            if mlflow is not None:
                mlflow.log_metric("validation/loss", val_loss, step=epoch_idx)

    return avg_loss, val_loss


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
@click.option(
    "--batch",
    type=click.INT,
    default=64,
    show_default=True,
    help="batch size for training",
)
@click.option("--patience", show_default=True, help="patience for early stopping")
@click.option("--delta", show_default=True, help="threshold to stop")
def train(
    epochs: int = 1,
    alpha: float = 0.001,
    batch: int = 64,
    patience: int = 3,
    delta: float = 0.01,
) -> dict[str, Any]:
    """Run SFT for a fixed number of epochs."""

    import mlflow

    mlflow.set_experiment("SFT Training")
    mlflow.config.enable_system_metrics_logging()  # pyright: ignore[reportPrivateImportUsage]
    mlflow.config.set_system_metrics_sampling_interval(1)  # pyright: ignore[reportPrivateImportUsage]

    with mlflow.start_run() as run:
        model, tokenizer = get_model()
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=alpha)
        tok_dataset = QwenTokDataset(dataset, tokenizer)
        dataloader = DataLoader(
            tok_dataset,
            batch_size=batch,
            shuffle=True,
            collate_fn=tok_dataset.collate_fn,
        )

        best_val_loss = float("inf")
        epochs_without_improve = 0
        early_stop_triggered = False

        payload = {
            "epochs": epochs,
            "learning rate": alpha,
            "batch size": batch,
            "model": f"qwen-{settings_conf.model}",
            "optimizer": type(optim).__name__,
            "loss": "CE Loss",
            "dataset size": len(tok_dataset),
        }
        if mlflow is not None:
            mlflow.log_params(payload)
        else:
            logger.info("sft_train_start", extra=payload)

        final_epoch_loss = 0.0
        final_val_loss: Optional[float] = None
        for ep in range(1, epochs + 1):
            train_loss, val_loss = epoch(model, dataloader, optim, ep)

            # INFO: guards for early stopping
            if val_loss is not None:
                if val_loss + delta < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

            if val_loss is not None and epochs_without_improve >= patience:
                early_stop_triggered = True
                msg = (
                    f"early stop triggered after {patience} epochs "
                    f"without improvement (val_loss={val_loss:.6f})"
                )
                logger.info(msg, extra={"epoch": ep, "patience": patience})
                if mlflow is not None:
                    mlflow.set_tag("error", msg)
                break

            final_epoch_loss = train_loss
            final_val_loss = val_loss
            if mlflow is not None:
                mlflow.log_metric("train/loss", train_loss, step=ep)
                if val_loss is not None:
                    mlflow.log_metric("validation/loss", val_loss, step=ep)

        summary = {
            "epochs": epochs,
            "learning_rate": alpha,
            "dataset_size": len(tok_dataset),
            "final_epoch_loss": final_epoch_loss,
            "final_val_loss": final_val_loss,
            "early_stopped": early_stop_triggered,
        }

        if mlflow is not None:
            mlflow.log_metrics(
                {
                    "summary/train_loss": final_epoch_loss,
                    "summary/val_loss": final_val_loss
                    if final_val_loss is not None
                    else -1.0,
                    "summary/early_stop": 1.0 if early_stop_triggered else 0.0,
                }
            )
            lora_artifact = Path("artifacts/lora_state.pt")
            lora_artifact.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), lora_artifact)
            mlflow.log_artifact(lora_artifact.as_posix())

        logger.info(
            "sft_train_complete epochs=%s final_epoch_loss=%.6f",
            epochs,
            final_epoch_loss,
        )
        return summary


if __name__ == "__main__":
    train()

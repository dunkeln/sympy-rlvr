"""One-off GSM8K benchmark for a PEFT-adapted Qwen model loaded from MLflow."""

from __future__ import annotations

import click
from tqdm import tqdm

from models.chat import generate_final_answer, get_model, normalize_answer
from sft.data_prep import load_gsm8k_train, transform_gsm8k_resp


@click.command()
@click.option("--run-id", required=True, help="MLflow run id with a logged LoRA adapter")
@click.option("--limit", type=click.INT, default=100, show_default=True, help="max GSM8K test rows to evaluate")
@click.option(
    "--max-new-tokens",
    type=click.INT,
    default=256,
    show_default=True,
    help="max tokens to generate per GSM8K question",
)
def bench(run_id: str, limit: int, max_new_tokens: int) -> None:
    """Benchmark exact-match GSM8K accuracy for one PEFT-adapted Qwen model."""
    model, tokenizer = get_model(run_id)
    dataset = load_gsm8k_train("test").map(transform_gsm8k_resp)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    correct = 0
    for row in tqdm(dataset, desc="gsm8k bench", unit="sample"):
        pred = generate_final_answer(
            model,
            tokenizer,
            question=str(row["question"]),
            max_new_tokens=max_new_tokens,
        )
        gold = normalize_answer(str(row["final_answer"]))
        correct += int(pred == gold)

    accuracy = 100.0 * correct / len(dataset) if len(dataset) else 0.0
    print(f"GSM8K bench accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    bench()

from pathlib import Path
from datasets import Dataset


def load_rl_dataset(synth_path: str) -> Dataset:
    """Load verified synthesized questions for RL training.

    Expects columns: question, final_answer, verified.
    Filters to verified=True only.
    """
    path = Path(synth_path)
    if not path.exists():
        raise FileNotFoundError(f"Synthesized dataset not found at {path}")

    dataset = Dataset.from_parquet(str(path))
    dataset = dataset.filter(lambda r: r["verified"])
    return dataset.rename_column("final_answer", "ground_truth").select_columns(
        ["question", "ground_truth"]
    )

from pathlib import Path
import re
from xml.sax.saxutils import escape

from datasets import Dataset, load_dataset
from settings import get_logger

logger = get_logger(__name__)
_GSM8K_PARQUET_PATH = Path("data/gsm8k-main-train.parquet")


def load_gsm8k_train():
    try:
        dataset = Dataset.from_parquet(str(_GSM8K_PARQUET_PATH))
        logger.info(
            "Loaded GSM8K split=train from local parquet path=%s",
            _GSM8K_PARQUET_PATH,
        )
    except Exception as exc:
        logger.warning(
            "Local parquet unavailable at %s (%s). Fetching from Hugging Face.",
            _GSM8K_PARQUET_PATH,
            exc.__class__.__name__,
        )
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        _GSM8K_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(_GSM8K_PARQUET_PATH))
        logger.info(
            "Saved GSM8K split=train to local parquet path=%s", _GSM8K_PARQUET_PATH
        )

    logger.info("Loaded GSM8K split=train with num_rows=%s", len(dataset))
    return dataset


def transform_gsm8k_resp(row):
    raw_answer = row.get("answer", "")
    parts = raw_answer.split("####", maxsplit=1)
    reasoning_raw = parts[0].strip()
    final_raw = parts[1].strip() if len(parts) == 2 else ""

    # Preserve GSM8K calculator traces by converting <<...>> (and $<<...>>)
    # to inline
    # <calc>...</calc> tags inside reasoning.
    reasoning_with_calc = re.sub(
        r"\$?<<([^<>]+)>>",
        lambda m: f"<calc>{escape(m.group(1))}</calc>",
        reasoning_raw,
    )
    reasoning_clean = re.sub(r"\s+", " ", reasoning_with_calc).strip()

    # Keep only the trailing answer token (typically numeric) after ####.
    final_answer = final_raw.split()[0] if final_raw else ""
    final_answer = final_answer.replace(",", "")

    response_xml = (
        "<response>\n"
        "    <reasoning>\n"
        f"    {reasoning_clean}\n"
        "    </reasoning>\n"
        f"    <final_answer>{escape(final_answer)}</final_answer>\n"
        "</response>"
    )

    return {
        "reasoning": reasoning_clean,
        "final_answer": final_answer,
        "response_xml": response_xml,
        "answer": response_xml,
    }

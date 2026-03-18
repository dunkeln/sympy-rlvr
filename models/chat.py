"""Inference helpers for base and PEFT-augmented Qwen chat-style generation."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from peft import PeftModel

from models.base import get_model as get_base_model
from settings import get_logger


logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_lora_adapter(run_id: str) -> Path:
    """Resolve a locally logged LoRA adapter directory from an MLflow run id."""
    matches = list(Path("mlruns").glob(f"*/{run_id}/artifacts/lora_adapter"))
    if not matches:
        raise FileNotFoundError(f"Could not find local LoRA adapter for run_id={run_id}")
    return matches[0]


def get_model(run_id: str):
    """Load the configured base model, then attach a PEFT adapter from a run id."""
    get_base_model.cache_clear()
    model, tokenizer = get_base_model()
    adapter_dir = find_lora_adapter(run_id)
    logger.info("Attaching LoRA adapter run_id=%s path=%s", run_id, adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir).to(device)
    model.eval()
    return model, tokenizer


def extract_final_answer(text: str) -> str:
    """Extract the final answer from XML output or a fallback numeric suffix."""
    xml_match = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL)
    if xml_match:
        return normalize_answer(xml_match.group(1))

    number_matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if number_matches:
        return normalize_answer(number_matches[-1])
    return ""


def generate_final_answer(
    model: torch.nn.Module,
    tokenizer,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate one answer string and return the normalized final answer token."""
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_final_answer(generated_text)


def normalize_answer(text: str) -> str:
    """Normalize answers for exact-match GSM8K comparison."""
    return text.strip().replace(",", "")

import torch
import torch.nn.functional as F
from peft import PeftModel
from models.chat import get_model
from verifier.reward import reward
from accelerate import Accelerator
import click

accelerator = Accelerator()


def rollout(
    run_id: str,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    G: int = 8,
):
    model, tokenizer = get_model(run_id)
    inputs = tokenizer(question, return_tensors="pt").to(accelerator.device)
    input_ids = inputs["input_ids"].repeat(G, 1)
    attention_mask = inputs["attention_mask"].repeat(G, 1)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[-1]
    return [
        tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
        for i in range(G)
    ]

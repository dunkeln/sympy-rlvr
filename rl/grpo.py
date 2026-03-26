import torch
import torch.nn.functional as F
from models.chat import get_model
from verifier.reward import reward
from accelerate import Accelerator

accelerator = Accelerator()

# INFO:
# medium model: a1f89cd6481c42eca6725cd741f5b48a
# small model: f8da3e4ffdb94e85ac7d1912a49f7d45


def compute_log_probs(model, input_ids, gen_tokens, pad_id):
    """Forward pass to get log probs of gen_tokens given input_ids as context.

    Concatenates [prompt | completion], runs one forward pass, then extracts
    log probs at completion positions only. Shape: [G, completion_len].
    """
    full_ids = torch.cat([input_ids, gen_tokens], dim=1)          # [G, prompt+completion]
    attention_mask = (full_ids != pad_id).long()

    with torch.no_grad():
        logits = model(full_ids, attention_mask=attention_mask).logits  # [G, seq_len, vocab]

    # shift: logits[t] predicts token[t+1], so completion logits start at prompt_len-1
    prompt_len = input_ids.shape[1]
    completion_logits = logits[:, prompt_len - 1:-1, :]            # [G, completion_len, vocab]
    log_probs = F.log_softmax(completion_logits, dim=-1)           # [G, completion_len, vocab]
    token_log_probs = log_probs.gather(
        dim=2,
        index=gen_tokens.unsqueeze(-1)
    ).squeeze(-1)                                                   # [G, completion_len]

    mask = (gen_tokens != pad_id).float()
    return token_log_probs * mask                                   # zero out pad positions


def grpo(run_id: str):
    model, tokenizer = get_model(run_id)
    model = model.to(accelerator.device)

    ref_model, _ = get_model(run_id)
    ref_model = ref_model.to(accelerator.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def rollout(
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        G: int = 8,
    ):
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
                pad_token_id=pad_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        seqs = outputs.sequences
        scores = outputs.scores
        prompt_len = input_ids.shape[-1]
        gen_tokens = seqs[:, prompt_len:]                          # [G, completion_len]

        # old policy log probs from generation scores
        step_log_probs = []
        for tok_idx, step_logits in enumerate(scores):
            step_logprobs = torch.log_softmax(step_logits, dim=-1)
            token_ids = gen_tokens[:, tok_idx]
            idx = torch.arange(step_logprobs.size(0), device=step_logprobs.device)
            step_log_probs.append(step_logprobs[idx, token_ids])

        old_log_probs = torch.stack(step_log_probs, dim=1)        # [G, completion_len]
        mask = (gen_tokens != pad_id).float()
        old_log_probs = old_log_probs * mask                       # zero out pad positions

        # reference model log probs
        ref_log_probs = compute_log_probs(ref_model, input_ids, gen_tokens, pad_id)

        texts = [
            tokenizer.decode(gen_tokens[i], skip_special_tokens=True)
            for i in range(G)
        ]

        return texts, gen_tokens, old_log_probs, ref_log_probs, input_ids

    return rollout


if __name__ == "__main__":
    rollouts = grpo("f8da3e4ffdb94e85ac7d1912a49f7d45")
    resps = rollouts("what is 2 + 2?")
    for idx, resp in enumerate(resps):
        print(f"turn {idx + 1}")
        print(resp, end="\n\n")

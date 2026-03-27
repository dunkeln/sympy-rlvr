import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.chat import get_model
from verifier.reward import reward
from rl.data_prep import load_rl_dataset
from accelerate import Accelerator
import click

try:
    import mlflow
except ImportError:
    mlflow = None

from settings import get_logger

logger = get_logger(__name__)
accelerator = Accelerator()

# INFO:
# medium model: a1f89cd6481c42eca6725cd741f5b48a
# small model: f8da3e4ffdb94e85ac7d1912a49f7d45


# --- log probs ---


def compute_log_probs(model, input_ids, gen_tokens, pad_id):
    """Forward pass → per-token log probs for completion tokens only.

    No torch.no_grad() here — caller controls gradient context.
    Shape: [G, completion_len].
    """
    full_ids = torch.cat([input_ids, gen_tokens], dim=1)
    attention_mask = (full_ids != pad_id).long()
    logits = model(full_ids, attention_mask=attention_mask).logits  # [G, seq, vocab]
    prompt_len = input_ids.shape[1]
    completion_logits = logits[:, prompt_len - 1 : -1, :]  # [G, completion_len, vocab]
    log_probs = F.log_softmax(completion_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=2, index=gen_tokens.unsqueeze(-1)).squeeze(
        -1
    )  # [G, completion_len]
    mask = (gen_tokens != pad_id).float()
    return token_log_probs * mask


# --- advantages ---


def compute_advantages(rewards: list[float]) -> torch.Tensor:
    r = torch.tensor(rewards, dtype=torch.float32, device=accelerator.device)
    return (r - r.mean()) / (r.std() + 1e-8)  # [G]


# --- loss ---


def grpo_loss(
    cur_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
) -> torch.Tensor:
    adv = advantages.unsqueeze(1)  # [G, 1]
    ratio = torch.exp(cur_log_probs - old_log_probs)  # [G, completion_len]
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * adv, clipped * adv)  # [G, completion_len]
    kl = cur_log_probs - ref_log_probs  # [G, completion_len]
    return (policy_loss + kl_beta * kl).mean()


# --- rollout ---


def grpo(run_id: str):
    model, tokenizer = get_model(run_id)
    model = model.to(accelerator.device)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)

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
            )

        seqs = outputs
        prompt_len = input_ids.shape[-1]
        gen_tokens = seqs[:, prompt_len:]  # [G, completion_len]

        with torch.no_grad():
            old_log_probs = compute_log_probs(model, input_ids, gen_tokens, pad_id)
            ref_log_probs = compute_log_probs(ref_model, input_ids, gen_tokens, pad_id)

        texts = [
            tokenizer.decode(gen_tokens[i], skip_special_tokens=True) for i in range(G)
        ]

        return texts, gen_tokens, old_log_probs, ref_log_probs, input_ids

    return model, tokenizer, ref_model, pad_id, rollout


# --- training loop ---


@click.command()
@click.option("--run-id", required=True, help="MLflow run id of the SFT checkpoint")
@click.option("--epochs", default=1, show_default=True)
@click.option("--alpha", default=1e-5, show_default=True, help="learning rate")
@click.option("--G", "g", default=8, show_default=True, help="completions per question")
@click.option("--clip-eps", default=0.2, show_default=True)
@click.option("--kl-beta", default=0.01, show_default=True)
@click.option("--max-new-tokens", default=512, show_default=True)
@click.option("--temperature", default=0.8, show_default=True)
@click.option("--synth-path", required=True, help="path to verified synthesized parquet")
def train(run_id, epochs, alpha, g, clip_eps, kl_beta, max_new_tokens, temperature, synth_path):
    model, tokenizer, ref_model, pad_id, rollout = grpo(run_id)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=alpha)

    dataset = load_rl_dataset(synth_path=synth_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    if mlflow:
        mlflow.log_params(
            {
                "run_id": run_id,
                "epochs": epochs,
                "lr": alpha,
                "G": g,
                "clip_eps": clip_eps,
                "kl_beta": kl_beta,
                "synth_path": synth_path,
                "dataset_size": len(dataset),
            }
        )

    global_step = 0
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        with tqdm(dataloader, desc=f"epoch {ep}", unit="q") as bar:
            for batch in bar:
                question = batch["question"][0]
                ground_truth = batch["ground_truth"][0]

                # rollout — eval mode so dropout is off during generation
                model.eval()
                texts, gen_tokens, old_log_probs, ref_log_probs, input_ids = rollout(
                    question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    G=g,
                )
                model.train()

                # score
                rewards = [reward(t, ground_truth, question) for t in texts]
                advantages = compute_advantages(rewards)

                # current policy log probs (with gradients)
                cur_log_probs = compute_log_probs(model, input_ids, gen_tokens, pad_id)

                # loss + update
                loss = grpo_loss(
                    cur_log_probs,
                    old_log_probs,
                    ref_log_probs,
                    advantages,
                    clip_eps,
                    kl_beta,
                )
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optim.step()

                loss_val = float(loss.detach().item())
                mean_reward = sum(rewards) / len(rewards)
                epoch_loss += loss_val
                global_step += 1

                bar.set_postfix(loss=f"{loss_val:.4f}", reward=f"{mean_reward:.2f}")

                if mlflow and global_step % 25 == 0:
                    mlflow.log_metrics(
                        {
                            "train/loss": loss_val,
                            "train/mean_reward": mean_reward,
                            "train/advantages_std": float(advantages.std().item()),
                            "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                            "gpu/memory_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                        },
                        step=global_step,
                    )

        avg_loss = epoch_loss / len(dataloader)
        logger.info("grpo epoch=%d avg_loss=%.6f", ep, avg_loss)
        if mlflow:
            mlflow.log_metric("epoch/loss", avg_loss, step=ep)

    # save checkpoint
    adapter_dir = Path("artifacts/lora_adapter")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    if mlflow:
        mlflow.log_artifacts(adapter_dir.as_posix(), artifact_path="lora_adapter")
    logger.info("saved RL adapter to %s", adapter_dir)


if __name__ == "__main__":
    import mlflow as _mlflow

    _mlflow.set_experiment("GRPO Training")
    _mlflow.config.enable_system_metrics_logging()
    _mlflow.config.set_system_metrics_sampling_interval(1)
    _mlflow.start_run()
    try:
        train()
        _mlflow.end_run(status="FINISHED")
    except Exception:
        _mlflow.end_run(status="FAILED")
        raise

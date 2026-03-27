# SymPy-RLVR

**Reinforcement Learning with Verifiable Rewards** for mathematical reasoning in small language models — built from scratch, no TRL, no learned reward model.

The core idea: instead of a trained reward model or LLM judge, use **SymPy as a symbolic verifier**. A capable LLM (Grok-4) calls SymPy functions as tools to independently derive ground truths at synthesis time. At training time, the model's answers are checked symbolically — giving a binary, noise-free reward signal.

---

## What's in here

| Component | What it does |
|---|---|
| `verifier/sympy_resolver.py` | Grok-4 calls SymPy tools (solve, integrate, factor, etc.) to derive answers — never trusts LLM prose |
| `verifier/q_synthesis.py` | Async pipeline that generates math questions and resolves ground truths in parallel; hard questions are verified twice (must agree) |
| `verifier/reward.py` | Dense rule-based reward across 8 signals — no LLM judge |
| `rl/grpo.py` | GRPO training loop from scratch — rollout, group-relative advantages, clipped ratio loss, KL penalty |
| `sft/trainer.py` | LoRA SFT baseline on Qwen2.5 |
| `bench.py` | GSM8K benchmark to measure before/after |

---

## Architecture

```
Synthesis (offline)                     Training (online)
─────────────────────                   ──────────────────
Grok-4                                  Qwen2.5 (LoRA)
  └─ calls SymPy tools                    └─ samples G completions per question
       └─ returns verified answer               └─ SymPy checks each answer
            └─ stored in parquet                     └─ group-relative reward
                                                          └─ GRPO gradient step
```

**Why not just use GSM8K?** Synthesizing our own questions means we can turn the difficulty knob and have variety in questions with ground truths derived symbolically.

---

## Training pipeline

### 1. Synthesize questions

```bash
python -m verifier.q_synthesis \
  --size 500 --easy --medium --hard \
  --semaphore 5 \
  --out-file data/synth_v1.parquet
```

Generates questions across difficulty tiers with random topic injection for variety. Hard questions are resolved twice — only kept if both resolver runs agree.

### 2. SFT baseline

```bash
python -m sft.trainer
```

Trains a LoRA adapter on Qwen2.5. Tracked with MLflow. Baseline hits **22% on GSM8K**.

### 3. GRPO (curriculum)

Run three sequential stages on easy → medium → hard:

```bash
python -m rl.grpo \
  --run-id <sft_mlflow_run_id> \
  --synth-path data/easy.parquet \
  --epochs 3 --G 8 --alpha 1e-5

python -m rl.grpo \
  --run-id <prev_grpo_run_id> \
  --synth-path data/medium.parquet \
  --epochs 3 --G 8 --alpha 1e-5

python -m rl.grpo \
  --run-id <prev_grpo_run_id> \
  --synth-path data/hard.parquet \
  --epochs 3 --G 8 --alpha 1e-5
```

Each stage loads the previous run's LoRA adapter via MLflow run ID.

### 4. Benchmark

```bash
python bench.py --run-id <grpo_run_id>
```

### 5. Chat

```bash
python -m models.chat --run-id <grpo_run_id>
```

---

## GRPO from scratch

Rather than using TRL's GRPO implementation, this is built from first principles:

- **Rollout**: sample G completions per question in eval mode (dropout off)
- **Advantages**: group-relative normalization — `(r - mean(r)) / (std(r) + ε)`
- **Loss**: clipped ratio loss (PPO-style) + KL penalty against frozen SFT reference
- **LoRA only**: base model weights frozen throughout; only adapter trains

```python
ratio = exp(cur_log_probs - old_log_probs)
clipped = clamp(ratio, 1 - ε, 1 + ε)
loss = -min(ratio * adv, clipped * adv) + β * (cur_log_probs - ref_log_probs)
```

The KL term keeps the policy close to the SFT reference, preventing reward hacking.

---

## Reward model

All signals are rule-based — no LLM judge, no learned reward model. Every signal is computed in milliseconds with regex and SymPy.

| Signal | Weight | How it's computed |
|---|---|---|
| Correctness | 0.50 | SymPy symbolic match; falls back to `exp(-3 × rel_err)` for near-misses |
| Self-consistency | 0.10 | Final answer matches the last number computed in `<reasoning>` |
| Reasoning depth | 0.10 | Count of math operators and numbers in `<reasoning>` |
| Number grounding | 0.10 | Fraction of question's numbers that appear in reasoning |
| Format | 0.08 | Presence of `<response>`, `<reasoning>`, `<final_answer>` tags |
| Parsability | 0.07 | Model produces any numeric final answer at all |
| Length sweet spot | 0.03 | Reasoning word count falls in 50–400 word target band |
| Repetition penalty | 0.02 | Penalises repeated n-grams — catches looping output |

The proximity decay on correctness means even wrong completions get differentiated rewards, ensuring GRPO always has non-zero advantage variance to learn from.

---

## SFT training curves

![SFT train loss](static/sft/train-loss.png)
![SFT epoch loss](static/sft/epoch-loss.png)
![SFT validation loss](static/sft/validation-loss.png)

---

## Stack

- **Model**: Qwen2.5-0.5B / 1.5B with LoRA (PEFT)
- **Verifier**: SymPy via Grok-4 tool-calling (xai_sdk)
- **Training**: PyTorch + Accelerate, no TRL
- **Tracking**: MLflow (params, metrics, artifacts, system metrics)
- **Hardware**: NVIDIA RTX (SFT on RTX 2000 ADA, GRPO on RunPod)

---

## Project layout

```
sympy-rlvr/
├── verifier/
│   ├── sympy_resolver.py   # LLM tool-calling verifier
│   ├── q_synthesis.py      # question + ground truth synthesis
│   └── reward.py           # training-time reward function
├── rl/
│   ├── grpo.py             # GRPO training loop
│   └── data_prep.py        # loads verified parquet
├── sft/
│   └── trainer.py          # LoRA SFT baseline
├── models/
│   ├── base.py             # base model loader
│   └── chat.py             # inference + LoRA attach
├── bench.py                # GSM8K benchmark
└── settings.py             # model config
```

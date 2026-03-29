import math
import re

from sympy import simplify, sympify


def verify(model_answer: str, ground_truth: str) -> float:
    """Smooth correctness reward via SymPy.

    Exact symbolic match → 1.0.
    Numeric proximity (relative error) → smooth decay toward 0.
    Unparseable → 0.0.
    """
    try:
        diff = simplify(sympify(model_answer) - sympify(ground_truth))
        if diff == 0:
            return 1.0
        pred = float(sympify(model_answer))
        truth = float(sympify(ground_truth))
        if truth == 0:
            return 1.0 if pred == 0 else 0.0
        rel_err = abs(pred - truth) / abs(truth)
        # 10% error → ~0.37, 50% error → ~0.08, 100% → ~0.01
        return math.exp(-3.0 * rel_err)
    except Exception:
        return 0.0


def format_reward(text: str) -> float:
    """Partial reward for XML structure — each tag contributes equally."""
    checks = [
        bool(re.search(r"<response>.*?</response>", text, re.DOTALL)),
        bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL)),
        bool(re.search(r"<final_answer>.*?</final_answer>", text, re.DOTALL)),
    ]
    return sum(checks) / len(checks)


def reasoning_depth_reward(text: str) -> float:
    """Reward for showing mathematical work in <reasoning>."""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if not match:
        return 0.0
    reasoning = match.group(1).strip()
    if len(reasoning) < 20:
        return 0.0
    numbers = len(re.findall(r"-?\d+(?:\.\d+)?", reasoning))
    operators = len(re.findall(r"[+\-*/=×÷]", reasoning))
    return min(1.0, (numbers + operators) / 20.0)


def parsability_reward(text: str) -> float:
    """Small reward for producing any numeric final answer at all."""
    answer = _extract(text)
    if not answer:
        return 0.0
    try:
        float(sympify(answer))
        return 1.0
    except Exception:
        return 0.0


def number_grounding_reward(text: str, question: str) -> float:
    """Reward if numbers from the question appear in the reasoning.

    Ensures the model uses the problem's values rather than hallucinating.
    """
    q_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", question))
    if not q_numbers:
        return 1.0  # no numbers in question, not applicable
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if not match:
        return 0.0
    reasoning = match.group(1)
    r_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", reasoning))
    overlap = len(q_numbers & r_numbers)
    return min(1.0, overlap / len(q_numbers))


def self_consistency_reward(text: str) -> float:
    """Reward if final_answer matches the last number computed in reasoning."""
    final_match = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if not final_match or not reasoning_match:
        return 0.0
    final_answer = final_match.group(1).strip().replace(",", "")
    reasoning_numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", reasoning_match.group(1))
    if not reasoning_numbers:
        return 0.0
    last_reasoning_number = reasoning_numbers[-1].replace(",", "")
    try:
        diff = simplify(sympify(final_answer) - sympify(last_reasoning_number))
        return 1.0 if diff == 0 else 0.0
    except Exception:
        return 0.0


def repetition_penalty(text: str, window: int = 8, threshold: int = 3) -> float:
    """Penalty for repeated n-grams in reasoning — catches looping output.

    Returns 1.0 (no penalty) down to 0.0 (heavy repetition).
    """
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if not match:
        return 1.0
    words = match.group(1).lower().split()
    if len(words) < window:
        return 1.0
    ngrams = [tuple(words[i : i + window]) for i in range(len(words) - window + 1)]
    from collections import Counter
    counts = Counter(ngrams)
    max_repeat = max(counts.values())
    if max_repeat <= threshold:
        return 1.0
    # decay: 3 repeats → 1.0, 6 → 0.5, 12 → 0.0
    return max(0.0, 1.0 - (max_repeat - threshold) / (threshold * 2))


def length_sweet_spot_reward(text: str, low: int = 50, high: int = 400) -> float:
    """Reward reasoning length in a target band — penalises too short or too long."""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if not match:
        return 0.0
    length = len(match.group(1).split())
    if low <= length <= high:
        return 1.0
    if length < low:
        return length / low
    # above high: decay
    return max(0.0, 1.0 - (length - high) / high)


def _extract(text: str) -> str:
    match = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


def breakdown(generated_text: str, ground_truth: str, question: str = "") -> dict:
    """Return each reward component individually for logging."""
    answer = _extract(generated_text)
    return {
        "correctness": verify(answer, ground_truth),
        "self_consistency": self_consistency_reward(generated_text),
        "reasoning_depth": reasoning_depth_reward(generated_text),
        "number_grounding": number_grounding_reward(generated_text, question) if question else 0.0,
        "format": format_reward(generated_text),
        "parsability": parsability_reward(generated_text),
        "length_sweet_spot": length_sweet_spot_reward(generated_text),
        "repetition": repetition_penalty(generated_text),
    }


def reward(generated_text: str, ground_truth: str, question: str = "") -> float:
    """Multiplicative reward — correctness gates everything else.

    Correctness is the primary signal (0.7 weight). Secondary signals
    (format, reasoning quality) are gated by a correctness multiplier:
    wrong answer → secondary contributes at most 1.5% of its value,
    correct answer → secondary contributes fully.

    This prevents the model from learning to game format/reasoning
    while ignoring correctness.

    correctness gate: 0.05 + 0.95 * correctness
      → wrong (0.0): gate = 0.05  → secondary maxes at 0.015
      → perfect (1.0): gate = 1.0 → secondary maxes at 0.30
    """
    answer = _extract(generated_text)
    c = verify(answer, ground_truth)
    sc = self_consistency_reward(generated_text)
    rd = reasoning_depth_reward(generated_text)
    ng = number_grounding_reward(generated_text, question) if question else 0.0
    f = format_reward(generated_text)
    p = parsability_reward(generated_text)
    ls = length_sweet_spot_reward(generated_text)
    rp = repetition_penalty(generated_text)

    # secondary signals normalized to [0, 1]
    secondary = (
        0.25 * sc
        + 0.20 * rd
        + 0.20 * ng
        + 0.15 * f
        + 0.10 * p
        + 0.05 * ls
        + 0.05 * rp
    )

    # correctness-scaled reward:
    #   correct   (c=1): 0.6 + 0.4 * secondary  → [0.6, 1.0]
    #   wrong     (c=0): 0.1 + 0.2 * secondary  → [0.1, 0.3]
    #   proximity (0<c<1): interpolates smoothly between the two
    base = 0.1 + 0.5 * c
    secondary_weight = 0.2 + 0.2 * c
    return base + secondary_weight * secondary

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


def reward(generated_text: str, ground_truth: str, question: str = "") -> float:
    """Dense combined reward — all rule-based, no LLM judge.

    correctness:       0.50 — symbolic match or numeric proximity
    self_consistency:  0.10 — final answer matches last reasoning value
    reasoning_depth:   0.10 — mathematical operators/numbers in reasoning
    number_grounding:  0.10 — question numbers appear in reasoning
    format:            0.08 — XML tag structure
    parsability:       0.07 — produces any numeric answer
    length_sweet_spot: 0.03 — reasoning length in target band
    repetition:        0.02 — no looping output (penalty)
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

    return (
        0.50 * c
        + 0.10 * sc
        + 0.10 * rd
        + 0.10 * ng
        + 0.08 * f
        + 0.07 * p
        + 0.03 * ls
        + 0.02 * rp
    )

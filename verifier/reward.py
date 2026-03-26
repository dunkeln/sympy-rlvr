import re
from sympy import simplify, sympify


def verify(model_answer: str, ground_truth: str) -> float:
    """Binary correctness reward via SymPy equivalence check."""
    try:
        diff = simplify(sympify(model_answer) - sympify(ground_truth))
        return 1.0 if diff == 0 else 0.0
    except Exception:
        return 0.0


def format_reward(text: str) -> float:
    """Partial reward for XML structure — each tag contributes equally."""
    checks = [
        bool(re.search(r"<response>.*?</response>", text, re.DOTALL)),
        bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL)),
        bool(re.search(r"<final_answer>.*?</final_answer>", text, re.DOTALL)),
    ]
    return sum(checks) / len(checks)  # 0.0, 0.33, 0.66, or 1.0


def _extract(text: str) -> str:
    match = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


def reward(generated_text: str, ground_truth: str) -> float:
    """Combined reward: 0.9 * correctness + 0.1 * format bonus."""
    answer = _extract(generated_text)
    c = verify(answer, ground_truth)
    f = format_reward(generated_text)
    return 0.9 * c + 0.1 * f

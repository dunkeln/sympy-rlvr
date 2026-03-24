import os
import json
import asyncio
from sympy import symbols, solve, sympify, diff, integrate, simplify, binomial, factorial, fibonacci
from sympy.ntheory import isprime, factorint, totient, nextprime, primerange
from sympy.stats import E, P, Normal, Binomial as BinomialDist, variance
from pathlib import Path
from dotenv import load_dotenv

from xai_sdk import AsyncClient as Client
from xai_sdk.chat import system, user, tool, tool_result

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")


class EvaluateArgs(BaseModel):
    expression: str = Field(
        description="A SymPy-parseable arithmetic expression e.g. '3*4 + 7'"
    )


class SolveArgs(BaseModel):
    equations: list[str] = Field(
        description="Equations as expressions equal to zero e.g. ['3*x + 7 - 22']"
    )
    solve_for: str = Field(description="Variable name to solve for e.g. 'x'")


class DiffArgs(BaseModel):
    expression: str
    variable: str


class IntegrateArgs(BaseModel):
    expression: str
    variable: str


class SimplifyArgs(BaseModel):
    expression: str


class CombinatorialArgs(BaseModel):
    n: int = Field(description="Total items")
    k: int = Field(description="Items chosen")


class FactorArgs(BaseModel):
    n: int = Field(description="Integer to factorize")


class NthPrimeArgs(BaseModel):
    n: int = Field(description="Find the next prime after this number")


class PrimeRangeArgs(BaseModel):
    a: int = Field(description="Start of range (inclusive)")
    b: int = Field(description="End of range (exclusive)")


class NormalDistArgs(BaseModel):
    mean: str = Field(description="Mean of the distribution as a SymPy expression")
    std: str = Field(description="Standard deviation as a SymPy expression")
    query: str = Field(description="Expression to compute E(...) or P(...) e.g. 'X > 2'")


def _sympy_evaluate(expression: str) -> str:
    try:
        return str(sympify(expression))
    except Exception as e:
        return f"error: {e}"


def _sympy_solve(equations: list[str], solve_for: str) -> str:
    try:
        sym = symbols(solve_for)
        exprs = [sympify(eq) for eq in equations]
        return str(solve(exprs, sym))
    except Exception as e:
        return f"error: {e}"


def _sympy_diff(expression: str, variable: str) -> str:
    try:
        return str(diff(sympify(expression), symbols(variable)))
    except Exception as e:
        return f"error: {e}"


def _sympy_integrate(expression: str, variable: str) -> str:
    try:
        return str(integrate(sympify(expression), symbols(variable)))
    except Exception as e:
        return f"error: {e}"


def _sympy_simplify(expression: str) -> str:
    try:
        return str(simplify(sympify(expression)))
    except Exception as e:
        return f"error: {e}"


def _sympy_combinations(n: int, k: int) -> str:
    try:
        return str(binomial(n, k))
    except Exception as e:
        return f"error: {e}"


def _sympy_permutations(n: int, k: int) -> str:
    try:
        return str(factorial(n) // factorial(n - k))
    except Exception as e:
        return f"error: {e}"


def _sympy_factorize(n: int) -> str:
    try:
        return str(factorint(n))
    except Exception as e:
        return f"error: {e}"


def _sympy_is_prime(n: int) -> str:
    try:
        return str(isprime(n))
    except Exception as e:
        return f"error: {e}"


def _sympy_totient(n: int) -> str:
    try:
        return str(totient(n))
    except Exception as e:
        return f"error: {e}"


def _sympy_next_prime(n: int) -> str:
    try:
        return str(nextprime(n))
    except Exception as e:
        return f"error: {e}"


def _sympy_primes_in_range(a: int, b: int) -> str:
    try:
        return str(list(primerange(a, b)))
    except Exception as e:
        return f"error: {e}"


def _sympy_fibonacci(n: int) -> str:
    try:
        return str(fibonacci(n))
    except Exception as e:
        return f"error: {e}"


TOOL_REGISTRY = {
    "sympy_evaluate":    lambda a: _sympy_evaluate(a["expression"]),
    "sympy_solve":       lambda a: _sympy_solve(a["equations"], a["solve_for"]),
    "sympy_diff":        lambda a: _sympy_diff(a["expression"], a["variable"]),
    "sympy_integrate":   lambda a: _sympy_integrate(a["expression"], a["variable"]),
    "sympy_simplify":    lambda a: _sympy_simplify(a["expression"]),
    "sympy_combinations":lambda a: _sympy_combinations(a["n"], a["k"]),
    "sympy_permutations":lambda a: _sympy_permutations(a["n"], a["k"]),
    "sympy_factorize":   lambda a: _sympy_factorize(a["n"]),
    "sympy_is_prime":    lambda a: _sympy_is_prime(a["n"]),
    "sympy_totient":     lambda a: _sympy_totient(a["n"]),
    "sympy_next_prime":  lambda a: _sympy_next_prime(a["n"]),
    "sympy_primes_in_range": lambda a: _sympy_primes_in_range(a["a"], a["b"]),
    "sympy_fibonacci":   lambda a: _sympy_fibonacci(a["n"]),
}

TOOLS = [
    tool(
        "sympy_evaluate",
        "Evaluate a pure arithmetic expression with no unknowns e.g. '3*4 + 7'",
        EvaluateArgs.model_json_schema(),
    ),
    tool(
        "sympy_solve",
        "Solve equations for a variable. Each equation must be an expression equal to zero e.g. '3*x+7-22' means 3x+7=22.",
        SolveArgs.model_json_schema(),
    ),
    tool(
        "sympy_diff",
        "Differentiate an expression with respect to a variable.",
        DiffArgs.model_json_schema(),
    ),
    tool(
        "sympy_integrate",
        "Indefinitely integrate an expression with respect to a variable.",
        IntegrateArgs.model_json_schema(),
    ),
    tool(
        "sympy_simplify",
        "Simplify a SymPy expression.",
        SimplifyArgs.model_json_schema(),
    ),
    # combinatorics
    tool(
        "sympy_combinations",
        "Compute C(n, k) — number of ways to choose k items from n without order.",
        CombinatorialArgs.model_json_schema(),
    ),
    tool(
        "sympy_permutations",
        "Compute P(n, k) — number of ordered arrangements of k items from n.",
        CombinatorialArgs.model_json_schema(),
    ),
    tool(
        "sympy_fibonacci",
        "Return the nth Fibonacci number.",
        FactorArgs.model_json_schema(),
    ),
    # number theory
    tool(
        "sympy_factorize",
        "Return the prime factorization of integer n as a dict {prime: exponent}.",
        FactorArgs.model_json_schema(),
    ),
    tool(
        "sympy_is_prime",
        "Return True if n is prime, False otherwise.",
        FactorArgs.model_json_schema(),
    ),
    tool(
        "sympy_totient",
        "Return Euler's totient φ(n) — count of integers up to n coprime to n.",
        FactorArgs.model_json_schema(),
    ),
    tool(
        "sympy_next_prime",
        "Return the smallest prime strictly greater than n.",
        NthPrimeArgs.model_json_schema(),
    ),
    tool(
        "sympy_primes_in_range",
        "Return all primes in [a, b).",
        PrimeRangeArgs.model_json_schema(),
    ),
]

SYSTEM_PROMPT = (
    "You are a precise math solver. "
    "Use the provided SymPy tools to compute every step — never guess or do arithmetic in your head. "
    "Break the problem into atomic steps and call a tool for each one. "
    "When you have the final answer, respond with just the value."
)


async def resolve(question: str, max_steps: int = 10) -> str:
    client = Client(api_key=os.getenv("XAI_API_KEY"))
    chat = client.chat.create(
        model="grok-4-fast-reasoning",
        messages=[system(SYSTEM_PROMPT), user(question)],
        tools=TOOLS,
    )

    for _ in range(max_steps):
        response = await chat.sample()

        if not response.tool_calls:
            return response.content.strip().splitlines()[0].strip()

        chat.append(response)

        for tc in response.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOL_REGISTRY[tc.function.name](args)
            chat.append(tool_result(result, tool_call_id=tc.id))

    return "unresolved"


if __name__ == "__main__":
    questions = [
        "What is 4 + 4 * 2?",
        "If 3x + 7 = 22, what is x?",
        "What is the derivative of x**3 + 2*x with respect to x?",
        "In how many ways can you choose 3 students from a class of 10?",
        "How many primes are there between 1 and 50?",
        "What is the 13th Fibonacci number?",
    ]

    async def main():
        for q in questions:
            print(f"Q: {q}")
            answer = await resolve(q)
            print(f"A: {answer}\n")

    asyncio.run(main())

import os
from sympy import symbols, solve, sympify
from pathlib import Path
from dotenv import load_dotenv
import asyncio

from xai_sdk import AsyncClient as Client
from xai_sdk.chat import system, user

from pydantic import BaseModel, Field
from typing import List

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")


class SymbolDef(BaseModel):
    name: str = Field(description="variable name, e.g. 'x'")
    assumptions: List[str] = Field(
        default=[],
        description="SymPy assumptions e.g. ['positive', 'integer']"
    )


class SymPyProblem(BaseModel):
    symbol_defs: List[SymbolDef] = Field(
        description="all unknown variables needed to express the equations"
    )
    equations: List[str] = Field(
        description=(
            "each equation as a SymPy-parseable expression equal to zero. "
            "e.g. '3*x + 7 - 22' means 3x+7=22. "
            "use standard Python math operators: *, **, /, +, -"
        )
    )
    solve_for: str = Field(
        description="the variable name to solve for"
    )


async def decomposer(question: str) -> SymPyProblem:
    client = Client(api_key=os.getenv("XAI_API_KEY"))
    chat = client.chat.create(
        model="grok-4-fast-reasoning",
        messages=[
            system(
                "You are a math-to-SymPy translator. "
                "Given a math problem, express it as a system of equations "
                "that SymPy can solve. All equations must be expressions equal to zero "
                "(move everything to one side). Use only standard arithmetic operators."
            ),
            user(question),
        ],
        response_format=SymPyProblem,
    )

    resp = await chat.parse(SymPyProblem)
    return resp[1]


def compile_and_solve(problem: SymPyProblem):
    sym_map = {
        s.name: symbols(s.name, **{a: True for a in s.assumptions})
        for s in problem.symbol_defs
    }

    exprs = [sympify(eq, locals=sym_map) for eq in problem.equations]
    target = sym_map[problem.solve_for]
    solution = solve(exprs, target)
    return solution


if __name__ == "__main__":
    question = "If 3x + 7 = 22, what is x?"

    async def main():
        problem = await decomposer(question)
        print("Structured problem:", problem.model_dump())
        solution = compile_and_solve(problem)
        print("Answer:", solution)

    asyncio.run(main())

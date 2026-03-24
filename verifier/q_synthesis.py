import os
import asyncio
import random
import click
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from settings import get_logger
from typing import Literal
from xai_sdk import AsyncClient
from xai_sdk.chat import system
from pydantic import BaseModel, Field

from verifier.sympy_resolver import resolve


logger = get_logger(__name__)

TOPICS = {
    "easy": [
        "basic arithmetic", "simple fractions", "percentages", "ratios",
        "linear equations", "area and perimeter", "unit conversion", "simple interest",
    ],
    "medium": [
        "quadratic equations", "systems of equations", "sequences and series",
        "exponential growth", "logarithms", "basic combinatorics", "probability",
        "polynomial arithmetic",
    ],
    "hard": [
        "number theory", "modular arithmetic", "generating functions",
        "recurrence relations", "complex numbers", "multivariable calculus",
        "Diophantine equations", "binomial theorem",
    ],
    "olympiad": [
        "modular arithmetic and residues", "combinatorial game theory",
        "analytic number theory", "algebraic number theory", "contour integration",
        "partition theory", "Ramanujan-style identities", "prime distribution",
    ],
}

load_dotenv(Path(".") / ".env")


class QuestionFormat(BaseModel):
    question: str = Field(description="question")
    difficulty: Literal["easy", "medium", "hard", "olympiad"] = Field(
        description="difficulty ranked in order of grade school math, college math, graduate level math and quant/olympiad difficulty math"
    )


def _chat(client: AsyncClient, difficulty: str, topic: str):
    return client.chat.create(
        model="grok-4-fast-non-reasoning",
        temperature=1.0,
        messages=[
            system(f"""
Generate a unique {difficulty} math problem focused on the topic: {topic}.
The answer must be a single exact numeric value (an integer, rational number,
or simplified expression like sqrt(2) or pi/4).

The problem must require computation, not proof. Stay within these categories:
algebra, polynomial equations, calculus (derivatives/integrals),
combinatorics (counting, permutations, combinations),
number theory (primes, divisibility, modular arithmetic),
or probability (expected value, simple distributions).

Do NOT generate:
- "Prove that..." problems
- Geometry construction or proof problems
- Problems whose answer is "infinitely many", "does not exist", or a set of values
- Problems requiring matrix operations or linear algebra

Be verbose and contextual in your phrasing as appropriate for {difficulty} level.
{"Focus on competition-style computation: modular arithmetic, combinatorial counts, closed-form sums, or exact integrals." if difficulty == "olympiad" else ""}
Do NOT include the answer.
            """),
        ],
        response_format=QuestionFormat,
    )


async def _generate_and_resolve(client: AsyncClient, difficulty: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        topic = random.choice(TOPICS[difficulty])
        _, q = await _chat(client, difficulty, topic).parse(QuestionFormat)
        logger.info("generated [%s/%s]: %s", difficulty, topic, q.question[:60])
        answer = await resolve(q.question)
        verified = answer != "unresolved"
        logger.info("resolved [%s]: %s", "ok" if verified else "fail", q.question[:60])
        return {
            "question": q.question,
            "difficulty": q.difficulty,
            "final_answer": answer if verified else None,
            "verified": verified,
        }


async def _create_tasks(size: int, easy, medium, hard, olympiad, semaphore, out_file):
    export_path = Path(".") / "data" / out_file
    if out_file and export_path.suffix != ".parquet":
        raise ValueError("File not a parquet file")

    difficulties = [
        d
        for d, flag in {
            "easy": easy,
            "medium": medium,
            "hard": hard,
            "olympiad": olympiad,
        }.items()
        if flag
    ]

    if not difficulties:
        raise ValueError("Select at least one difficulty")

    client = AsyncClient(api_key=os.getenv("XAI_API_KEY"))
    sem = asyncio.Semaphore(semaphore)

    logger.info("generating and resolving %d samples, difficulties: %s", size, difficulties)
    results = await asyncio.gather(*[
        _generate_and_resolve(client, random.choice(difficulties), sem)
        for _ in range(size)
    ])

    verified_count = sum(1 for r in results if r["verified"])
    logger.info("complete: %d/%d verified", verified_count, size)

    match out_file:
        case "":
            return results
        case _:
            pl.DataFrame(results).write_parquet(export_path)
            logger.info("saved to %s", export_path)


@click.command()
@click.option("--size", type=click.INT, default=10, help="number of tasks to generate")
@click.option("--easy", is_flag=True)
@click.option("--medium", is_flag=True)
@click.option("--hard", is_flag=True)
@click.option("--olympiad", is_flag=True)
@click.option("--semaphore", type=click.INT, default=3)
@click.option("--out-file", type=click.STRING, default="", help="export generations as parquet")
def create_tasks(size, easy, medium, hard, olympiad, semaphore, out_file):
    asyncio.run(_create_tasks(size, easy, medium, hard, olympiad, semaphore, out_file))


if __name__ == "__main__":
    create_tasks()

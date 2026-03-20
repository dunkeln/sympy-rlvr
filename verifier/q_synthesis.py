import os
import asyncio
import json
import random
import click
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from settings import get_logger
from typing import Literal
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user, assistant
from pydantic import BaseModel, Field


logger = get_logger(__name__)

load_dotenv(Path(".") / ".env")

logger.info("xai client loaded")


class QuestionFormat(BaseModel):
    question: str = Field(description="question")
    difficulty: Literal["easy", "medium", "hard", "olympiad"] = Field(
        description="difficulty ranked in order of grade school math, college math, graduate level math and quant/olympid difficulty math"
    )
    final_answer: int | float = Field(description="single numeric value final answer")


def chat(client: AsyncClient, difficulty):
    return client.chat.create(
        model="grok-4-fast-non-reasoning",
        messages=[
            system(f"""
            Generate a unique {difficulty} math problem. Be verbose with your phrasing as per the difficulty level.
            """),
        ],
        response_format=QuestionFormat,
    )


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

    async def run_one():
        async with sem:
            return await chat(client, random.choice(difficulties)).sample()

    questions = [run_one() for _ in range(size)]
    logger.info("generating %d samples, difficulties: %s", size, str(difficulties))
    questions = await asyncio.gather(*questions)
    logger.info("generation complete")
    questions = [json.loads(q.content) for q in questions]
    match out_file:
        case "":
            return questions
        case _:
            pl.DataFrame(questions).write_parquet(export_path)
            return


@click.command()
@click.option("--size", type=click.INT, default=10, help="number of tasks to generate")
@click.option("--easy", is_flag=True, help="generate hard questions")
@click.option("--medium", is_flag=True, help="generate hard questions")
@click.option("--hard", is_flag=True, help="generate hard questions")
@click.option("--olympiad", is_flag=True, help="generate hard questions")
@click.option("--semaphore", type=click.INT, default=3)
@click.option(
    "--out-file", type=click.STRING, default="", help="export generations as parquet"
)
def create_tasks(size: int, easy, medium, hard, olympiad, semaphore, out_file):
    return asyncio.run(
        _create_tasks(size, easy, medium, hard, olympiad, semaphore, out_file)
    )


if __name__ == "__main__":
    create_tasks()

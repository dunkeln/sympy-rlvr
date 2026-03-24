# TODO: build sympy decomposer and compiler
import os
from sympy import solve, symbols
from pathlib import Path
from dotenv import load_dotenv
import asyncio

from xai_sdk import AsyncClient as Client
from xai_sdk.chat import system, user

from pydantic import BaseModel, Field, RootModel
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")


class Symbol(RootModel[Dict[str, int]]):
    pass


class DecomposeState(BaseModel):
    ground_truths: str = Field(
        description="next step in a list that should assert towards the solution"
    )
    symbols: RootModel | None


async def decomposer(question: str):
    client = Client(api_key=os.getenv("XAI_API_KEY"))
    chat = client.chat.create(
        model="grok-4-fast-reasoning",
        messages=[
            system("define the first step(s) towards solving the problem."),
            user(question),
        ],
        response_format=DecomposeState,
    )

    resp = await chat.parse(DecomposeState)
    resp = resp[1].model_dump()
    return resp


if __name__ == "__main__":
    question = "what is 4 + 4 * 2?"

    resp = asyncio.run(decomposer(question))
    print(resp)

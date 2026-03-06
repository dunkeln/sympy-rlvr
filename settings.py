import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.logging import RichHandler

_DEFAULT_LEVEL: Final[str] = "INFO"


class MlflowMode(str, Enum):
    ON = "on"
    OFF = "off"


class ModelSize(str, Enum):
    SMALL = "small"
    MED = "med"
    LARGE = "large"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    set_mlflow: MlflowMode = MlflowMode.OFF
    model: ModelSize = ModelSize.MED

    @property
    def model_repo_id(self) -> str:
        model_map = {
            ModelSize.SMALL: "Qwen/Qwen2.5-0.5B",
            ModelSize.MED: "Qwen/Qwen2.5-1.5B",
            ModelSize.LARGE: "Qwen/Qwen2.5-3B",
        }
        return model_map[self.model]


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def _resolve_level() -> int:
    level_name = os.getenv("LOG_LEVEL", _DEFAULT_LEVEL).upper()
    level = logging.getLevelName(level_name)
    return level if isinstance(level, int) else logging.INFO


def configure_logging(*, force: bool = False) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers and not force:
        return

    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=False,
        tracebacks_show_locals=False,
    )

    logging.basicConfig(
        level=_resolve_level(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)

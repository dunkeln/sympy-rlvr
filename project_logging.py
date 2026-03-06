import logging
import os
from typing import Final

from rich.logging import RichHandler

_DEFAULT_LEVEL: Final[str] = "INFO"


def _resolve_level() -> int:
    level_name = os.getenv("LOG_LEVEL", _DEFAULT_LEVEL).upper()
    level = logging.getLevelName(level_name)
    return level if isinstance(level, int) else logging.INFO


def configure_logging(*, force: bool = False) -> None:
    """Configure root logging with a Rich handler."""
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
    """Return a named logger using the shared project configuration."""
    configure_logging()
    return logging.getLogger(name)

import logging

import pytest
from pydantic import ValidationError

from settings import AppSettings, ModelSize, _resolve_level


@pytest.mark.parametrize(
    ("size", "repo_id"),
    [
        (ModelSize.SMALL, "Qwen/Qwen2.5-0.5B"),
        (ModelSize.MED, "Qwen/Qwen2.5-1.5B"),
        (ModelSize.LARGE, "Qwen/Qwen2.5-3B"),
    ],
)
def test_model_repo_id_mapping(size: ModelSize, repo_id: str) -> None:
    settings = AppSettings(model=size)
    assert settings.model_repo_id == repo_id


def test_invalid_model_value_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        AppSettings(model="tiny")


def test_resolve_level_falls_back_to_info_for_invalid_level(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "not-a-level")
    assert _resolve_level() == logging.INFO

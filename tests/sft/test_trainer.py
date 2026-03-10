import types

import pytest

torch = pytest.importorskip("torch")
from datasets import Dataset

import sft.trainer as trainer


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool, return_tensors: str):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        tokens = [(ord(ch) % 31) + 1 for ch in text]
        if not tokens:
            tokens = [1]
        return {"input_ids": torch.tensor([tokens], dtype=torch.long)}


class DummyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        masked_labels = labels.masked_fill(labels == -100, 0)
        loss = (
            input_ids.float().mean()
            + masked_labels.float().mean()
            + attention_mask.float().mean()
        ) * self.scale
        return types.SimpleNamespace(loss=loss)


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3], [-100, 5, -100]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    }


def test_step_updates_model_params() -> None:
    model = DummyCausalLM().to(trainer.device)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    row = _make_batch()

    before = model.scale.detach().clone()
    loss = trainer.step(model, optim, row)

    assert isinstance(loss, float)
    assert model.scale.detach().item() != before.item()


def test_epoch_returns_average_loss() -> None:
    model = DummyCausalLM().to(trainer.device)
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    dataloader = [_make_batch(), _make_batch()]

    avg_loss, val_loss = trainer.epoch(model, dataloader, optim, epoch_idx=1)

    assert isinstance(avg_loss, float)
    assert avg_loss > 0.0
    assert val_loss is None


def test_epoch_returns_validation_loss() -> None:
    model = DummyCausalLM().to(trainer.device)
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    dataloader = [_make_batch()]
    val_dataloader = [_make_batch()]

    avg_loss, val_loss = trainer.epoch(
        model,
        dataloader,
        optim,
        epoch_idx=1,
        val_dataloader=val_dataloader,
    )

    assert isinstance(val_loss, float)
    assert val_loss > 0.0


def test_train_runs_with_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_dataset = Dataset.from_dict(
        {
            "question": ["ab", "cd"],
            "answer": ["xy", "zt"],
        }
    )

    monkeypatch.setattr(trainer, "dataset", tiny_dataset)
    monkeypatch.setattr(
        trainer,
        "get_model",
        lambda: (
            DummyCausalLM().to(trainer.device),
            FakeTokenizer(),
        ),
    )

    summary = trainer.train.callback(epochs=1, alpha=0.01)

    assert summary["epochs"] == 1
    assert summary["dataset_size"] == 2
    assert summary["final_epoch_loss"] > 0.0
    assert summary["final_val_loss"] is None

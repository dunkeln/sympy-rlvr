import pytest
torch = pytest.importorskip("torch")
from datasets import Dataset

from sft.qwen_tok_dataset import QwenTokDataset


class FakeTokenizer:
    eos_token = None
    pad_token = None
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool, return_tensors: str):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        tokens = [(ord(c) % 31) + 1 for c in text]
        if not tokens:
            tokens = [1]
        return {"input_ids": torch.tensor([tokens], dtype=torch.long)}


class FakeTokenizerWithoutPadId:
    eos_token = "<eos>"
    eos_token_id = 7
    pad_token = None
    pad_token_id = None

    def __call__(self, text: str, add_special_tokens: bool, return_tensors: str):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        tokens = [(ord(c) % 31) + 1 for c in text]
        if not tokens:
            tokens = [1]
        return {"input_ids": torch.tensor([tokens], dtype=torch.long)}


def test_qwen_tok_dataset_len() -> None:
    ds = Dataset.from_dict({"question": ["Q1", "Q2"], "answer": ["A1", "A2"]})
    tok_ds = QwenTokDataset(ds=ds, tok=FakeTokenizer())
    assert len(tok_ds) == 2


def test_qwen_tok_dataset_masks_question_tokens() -> None:
    ds = Dataset.from_dict({"question": ["ab"], "answer": ["xyz"]})
    tok_ds = QwenTokDataset(ds=ds, tok=FakeTokenizer())

    sample = tok_ds[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    q_len = 2
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert input_ids.shape == labels.shape
    assert torch.all(labels[:q_len] == -100)
    assert torch.equal(input_ids[q_len:], labels[q_len:].to(dtype=input_ids.dtype))


def test_qwen_tok_dataset_falls_back_to_eos_token_id_for_padding() -> None:
    ds = Dataset.from_dict({"question": ["ab"], "answer": ["xyz"]})
    tok_ds = QwenTokDataset(ds=ds, tok=FakeTokenizerWithoutPadId(), max_seq_len=8)

    batch = tok_ds.collate_fn([tok_ds[0]])

    assert tok_ds.tok.pad_token == tok_ds.tok.eos_token
    assert tok_ds.tok.pad_token_id == tok_ds.tok.eos_token_id
    assert batch["input_ids"].shape == (1, 8)

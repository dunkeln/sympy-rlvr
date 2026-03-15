"""Torch dataset wrapper for Qwen SFT tokenization.

Purpose:
- Convert Hugging Face rows into causal-LM tensors where question tokens are
  masked in labels and answer tokens are supervised.

If you need to change behavior:
- Update `__getitem__` to alter prompt/target composition or masking policy.
"""

from datasets import Dataset as HFDataset
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase


class QwenTokDataset(TorchDataset):
    """Provide tokenized question-answer samples for Qwen SFT.

    Owns:
    - Row-level tokenization and label masking strategy.

    Does not own:
    - Padding/collation or batching policy.

    Usage:
    - Instantiate with a transformed HF dataset and tokenizer.
    - Wrap with a DataLoader for training.
    """

    def __init__(self, ds: HFDataset, tok: PreTrainedTokenizerBase, max_seq_len=512):
        """Store dataset/tokenizer references used by sample conversion."""
        self.ds = ds
        self.tok = tok
        self.max_seq_len = max_seq_len

        eos_token = getattr(self.tok, "eos_token", None)
        if getattr(self.tok, "pad_token", None) is None and eos_token is not None:
            self.tok.pad_token = eos_token
        if getattr(self.tok, "pad_token_id", None) is None:
            eos_token_id = getattr(self.tok, "eos_token_id", None)
            if eos_token_id is not None:
                self.tok.pad_token_id = eos_token_id

    def __len__(self) -> int:
        """Return dataset size for DataLoader sampling."""
        return len(self.ds)

    def collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        B = len(batch)
        T = self.max_seq_len
        pad_id = self.tok.eos_token_id
        if not isinstance(pad_id, int):
            raise ValueError(
                "Tokenizer must expose an integer pad_token_id for collation"
            )
        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)

        for i, triplet in enumerate(batch):
            L = min(triplet["input_ids"].size(0), T)

            input_ids[i, :L] = triplet["input_ids"][:L]
            labels[i, :L] = triplet["labels"][:L]
            attention_mask[i, :L] = triplet["attention_mask"][:L]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        """Build one causal-LM sample with masked question labels.

        Args:
            index (int): Row index in the backing HF dataset.

        Returns:
            dict[str, torch.Tensor]: tokenized sample tensors for causal LM.
        """
        row = self.ds[index]
        question = str(row["question"])
        answer = str(row["answer"])

        eos_token = getattr(self.tok, "eos_token", None)
        if eos_token is not None:
            answer = f"{answer}{eos_token}"

        q_tokens = self.tok(question, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        a_tokens = self.tok(answer, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        # BUG: linter doesn't understand squeeze is right
        q_tokens, a_tokens = q_tokens.squeeze(0), a_tokens.squeeze(0)  # pyright: ignore[reportAttributeAccessIssue]

        input_ids = torch.cat([q_tokens, a_tokens], dim=0)

        labels = torch.cat(
            [
                torch.full((q_tokens.size(-1),), -100, dtype=torch.long),
                a_tokens.clone(),
            ],
            dim=0,
        )

        attn_mask = torch.ones(input_ids.size(-1), dtype=torch.long)

        assert (labels == -100).sum().item() == q_tokens.size(-1), (
            "labels tensors incorrectly padded"
        )
        assert input_ids.size() == labels.size(), "input dim != label dim"
        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "labels": labels,
            "attention_mask": attn_mask,
        }

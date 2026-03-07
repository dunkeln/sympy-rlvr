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

    def __init__(self, ds: HFDataset, tok: PreTrainedTokenizerBase):
        """Store dataset/tokenizer references used by sample conversion."""
        self.ds = ds
        self.tok = tok

    def __len__(self):
        """Return dataset size for DataLoader sampling."""
        return len(self.ds)

    def __getitem__(self, index):
        """Build one causal-LM sample with masked question labels.

        Args:
            index (int): Row index in the backing HF dataset.

        Returns:
            dict[str, torch.Tensor]: `input_ids` and `labels` tensors.
        """
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
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
            [torch.full((q_tokens.size(-1),), -100), a_tokens.clone()], dim=0
        )

        assert (labels == -100).sum().item() == q_tokens.size(-1), (
            "labels tensors incorrectly padded"
        )
        assert input_ids.size() == labels.size(), "input dim != label dim"
        return input_ids, labels

from pathlib import Path

import sft.data_prep as data_prep


class DummyDataset:
    def __init__(self, n: int) -> None:
        self.n = n
        self.saved_path: str | None = None

    def __len__(self) -> int:
        return self.n

    def to_parquet(self, path: str) -> None:
        self.saved_path = path


def test_load_gsm8k_train_prefers_local_parquet(monkeypatch, tmp_path: Path) -> None:
    local_path = tmp_path / "gsm8k-main-train.parquet"
    local_ds = DummyDataset(5)

    def fake_from_parquet(path: str):
        assert path == str(local_path)
        return local_ds

    def fake_load_dataset(*_args, **_kwargs):
        raise AssertionError("load_dataset should not be called when local parquet exists")

    monkeypatch.setattr(data_prep, "_GSM8K_PARQUET_PATH", local_path)
    monkeypatch.setattr(data_prep.Dataset, "from_parquet", staticmethod(fake_from_parquet))
    monkeypatch.setattr(data_prep, "load_dataset", fake_load_dataset)

    loaded = data_prep.load_gsm8k_train()
    assert loaded is local_ds


def test_load_gsm8k_train_fetches_and_saves_when_local_missing(
    monkeypatch, tmp_path: Path
) -> None:
    local_path = tmp_path / "data" / "gsm8k-main-train.parquet"
    remote_ds = DummyDataset(7)

    def fake_from_parquet(_path: str):
        raise OSError("missing local parquet")

    def fake_load_dataset(name: str, config: str, split: str):
        assert name == "openai/gsm8k"
        assert config == "main"
        assert split == "train"
        return remote_ds

    monkeypatch.setattr(data_prep, "_GSM8K_PARQUET_PATH", local_path)
    monkeypatch.setattr(data_prep.Dataset, "from_parquet", staticmethod(fake_from_parquet))
    monkeypatch.setattr(data_prep, "load_dataset", fake_load_dataset)

    loaded = data_prep.load_gsm8k_train()

    assert loaded is remote_ds
    assert remote_ds.saved_path == str(local_path)
    assert local_path.parent.exists()


def test_transform_gsm8k_resp_converts_calc_and_final_answer() -> None:
    sample = (
        "He sold each DVD for 6*2.5=$<<6*2.5=15>>15 "
        "Then after cost 450000-2000=$<<450000-2000=448000>>448,000 "
        "#### 448000"
    )

    out = data_prep.transform_gsm8k_resp({"answer": sample})

    assert out["final_answer"] == "448000"
    assert "<response>" in out["response_xml"]
    assert "<calc>6*2.5=15</calc>" in out["response_xml"]
    assert "<calc>450000-2000=448000</calc>" in out["response_xml"]
    assert out["answer"] == out["response_xml"]


def test_transform_gsm8k_resp_handles_missing_final_delimiter() -> None:
    out = data_prep.transform_gsm8k_resp({"answer": "Compute 1+1=<<1+1=2>>2"})

    assert out["final_answer"] == ""
    assert "<calc>1+1=2</calc>" in out["response_xml"]

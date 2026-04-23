import numpy as np
import json
import torch
from nash_llm.data.dataset import PretrainDataset


class TestPretrainDataset:
    def _make_dataset(self, tmp_path, n_tokens=1000, seq_len=64):
        tokens = np.random.randint(0, 50257, size=n_tokens, dtype=np.uint16)
        shard_path = tmp_path / "train_000.bin"
        tokens.tofile(str(shard_path))
        meta = {"vocab_size": 50257, "total_tokens": n_tokens}
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        return PretrainDataset(str(tmp_path), split="train", seq_len=seq_len)

    def test_len(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_tokens=1000, seq_len=64)
        assert len(ds) == 1000 // 64

    def test_getitem_shapes(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_tokens=1000, seq_len=64)
        x, y = ds[0]
        assert x.shape == (64,)
        assert y.shape == (64,)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

    def test_targets_shifted(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_tokens=1000, seq_len=64)
        x, y = ds[0]
        tokens = np.fromfile(str(tmp_path / "train_000.bin"), dtype=np.uint16)
        assert x[0].item() == tokens[0]
        assert y[0].item() == tokens[1]

    def test_getitem_reads_across_shard_boundary(self, tmp_path):
        first = np.arange(6, dtype=np.uint16)
        second = np.arange(6, 12, dtype=np.uint16)
        first.tofile(str(tmp_path / "train_000.bin"))
        second.tofile(str(tmp_path / "train_001.bin"))

        ds = PretrainDataset(str(tmp_path), split="train", seq_len=4)
        x, y = ds[1]

        assert x.tolist() == [4, 5, 6, 7]
        assert y.tolist() == [5, 6, 7, 8]

    def test_dataloader_compatible(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_tokens=1000, seq_len=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 64)
        assert batch_y.shape == (4, 64)

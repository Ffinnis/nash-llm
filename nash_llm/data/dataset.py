import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PretrainDataset(Dataset):
    def __init__(self, data_dir: str, split: str, seq_len: int):
        self.seq_len = seq_len
        data_path = Path(data_dir)

        shard_files = sorted(data_path.glob(f"{split}_*.bin"))
        if not shard_files:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}")

        arrays = []
        for shard in shard_files:
            arr = np.memmap(str(shard), dtype=np.uint16, mode="r")
            arrays.append(arr)

        self.data = np.concatenate(arrays)
        self.n_tokens = len(self.data)

    def __len__(self) -> int:
        return self.n_tokens // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = idx * self.seq_len
        x = self.data[offset : offset + self.seq_len].astype(np.int64)
        y = self.data[offset + 1 : offset + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

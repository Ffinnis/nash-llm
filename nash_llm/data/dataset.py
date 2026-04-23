import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from bisect import bisect_right


class PretrainDataset(Dataset):
    def __init__(self, data_dir: str, split: str, seq_len: int):
        self.seq_len = seq_len
        data_path = Path(data_dir)

        shard_files = sorted(data_path.glob(f"{split}_*.bin"))
        if not shard_files:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}")

        self.shards = []
        self.shard_ends: list[int] = []
        total_tokens = 0
        for shard in shard_files:
            arr = np.memmap(str(shard), dtype=np.uint16, mode="r")
            self.shards.append(arr)
            total_tokens += len(arr)
            self.shard_ends.append(total_tokens)

        self.n_tokens = total_tokens

    def __len__(self) -> int:
        return max((self.n_tokens - 1) // self.seq_len, 0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = idx * self.seq_len
        tokens = self._read_tokens(offset, self.seq_len + 1).astype(np.int64)
        return torch.from_numpy(tokens[:-1]), torch.from_numpy(tokens[1:])

    def _read_tokens(self, offset: int, length: int) -> np.ndarray:
        if offset < 0 or length < 0 or offset + length > self.n_tokens:
            raise IndexError("token window is out of bounds")

        out = np.empty(length, dtype=np.uint16)
        out_pos = 0
        shard_idx = bisect_right(self.shard_ends, offset)
        shard_start = self.shard_ends[shard_idx - 1] if shard_idx > 0 else 0
        local_offset = offset - shard_start

        while out_pos < length:
            shard = self.shards[shard_idx]
            take = min(length - out_pos, len(shard) - local_offset)
            out[out_pos : out_pos + take] = shard[local_offset : local_offset + take]
            out_pos += take
            shard_idx += 1
            local_offset = 0

        return out

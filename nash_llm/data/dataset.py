import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from bisect import bisect_right
import json


TOKEN_DTYPES = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
}


class PretrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        seq_len: int,
        expected_vocab_size: int | None = None,
        expected_representation: str | None = None,
    ):
        self.seq_len = seq_len
        data_path = Path(data_dir)
        self.meta = self._load_meta(data_path)
        self.token_dtype = self._resolve_token_dtype(self.meta)
        self._validate_meta(
            data_dir=data_dir,
            expected_vocab_size=expected_vocab_size,
            expected_representation=expected_representation,
        )

        shard_files = sorted(data_path.glob(f"{split}_*.bin"))
        if not shard_files:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}")

        self.shards = []
        self.shard_ends: list[int] = []
        total_tokens = 0
        for shard in shard_files:
            arr = np.memmap(str(shard), dtype=self.token_dtype, mode="r")
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

        out = np.empty(length, dtype=self.token_dtype)
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

    @staticmethod
    def _load_meta(data_path: Path) -> dict:
        meta_path = data_path / "meta.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text())

    @staticmethod
    def _resolve_token_dtype(meta: dict) -> np.dtype:
        dtype_name = meta.get("token_dtype", "uint16")
        if dtype_name not in TOKEN_DTYPES:
            raise ValueError(
                f"Unsupported token dtype '{dtype_name}' in dataset metadata. "
                f"Expected one of: {', '.join(TOKEN_DTYPES)}"
            )
        return np.dtype(TOKEN_DTYPES[dtype_name])

    def _validate_meta(
        self,
        data_dir: str,
        expected_vocab_size: int | None,
        expected_representation: str | None,
    ) -> None:
        if expected_vocab_size is not None and "vocab_size" in self.meta:
            actual_vocab_size = int(self.meta["vocab_size"])
            if actual_vocab_size != expected_vocab_size:
                raise ValueError(
                    f"Dataset vocab_size mismatch for {data_dir}: "
                    f"metadata has {actual_vocab_size}, config expects {expected_vocab_size}."
                )
        if expected_representation is not None and "representation" in self.meta:
            actual_representation = str(self.meta["representation"])
            if actual_representation != expected_representation:
                raise ValueError(
                    f"Dataset representation mismatch for {data_dir}: "
                    f"metadata has {actual_representation}, config expects {expected_representation}."
                )

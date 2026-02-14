import json
import torch
from torch.utils.data import Dataset
from nash_llm.data.tokenizer import Tokenizer


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer()
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        prompt_ids = self.tokenizer.encode(item["prompt"])
        completion_ids = self.tokenizer.encode(item["completion"])
        eot = [self.tokenizer.eot_token]

        full_ids = prompt_ids + completion_ids + eot
        full_ids = full_ids[: self.max_seq_len]

        prompt_len = min(len(prompt_ids), self.max_seq_len)
        seq_len = len(full_ids)

        loss_mask = [False] * prompt_len + [True] * (seq_len - prompt_len)

        pad_len = self.max_seq_len - seq_len
        input_ids = full_ids + [0] * pad_len
        targets = full_ids[1:] + [0] * (pad_len + 1)
        targets = targets[: self.max_seq_len]
        loss_mask = loss_mask + [False] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(targets, dtype=torch.int64),
            torch.tensor(loss_mask, dtype=torch.bool),
        )

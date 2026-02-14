import json
import torch
from nash_llm.data.sft_dataset import SFTDataset


class TestSFTDataset:
    def _make_dataset(self, tmp_path, max_seq_len=128):
        data = [
            {"prompt": "What is 2+2?", "completion": "4"},
            {"prompt": "Hello", "completion": "Hi there, how can I help?"},
        ]
        path = tmp_path / "sft_data.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return SFTDataset(str(path), max_seq_len=max_seq_len)

    def test_len(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        assert len(ds) == 2

    def test_getitem_returns_three_tensors(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        input_ids, targets, loss_mask = ds[0]
        assert input_ids.dtype == torch.int64
        assert targets.dtype == torch.int64
        assert loss_mask.dtype == torch.bool

    def test_loss_mask_masks_prompt(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        input_ids, targets, loss_mask = ds[0]
        assert not loss_mask[0].item()  # prompt start is masked
        assert loss_mask.any()  # at least some completion tokens unmasked

    def test_padding_to_max_seq_len(self, tmp_path):
        ds = self._make_dataset(tmp_path, max_seq_len=128)
        input_ids, targets, loss_mask = ds[0]
        assert input_ids.shape == (128,)
        assert targets.shape == (128,)
        assert loss_mask.shape == (128,)

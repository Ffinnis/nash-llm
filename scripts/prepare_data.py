"""Download and tokenize a dataset into binary shards for pretraining."""
import argparse
import json
import numpy as np
from pathlib import Path
from nash_llm.data.tokenizer import Tokenizer

DATASET_CONFIGS = {
    "tinystories_10M": {"hf_path": "roneneldan/TinyStories", "split": "train", "max_tokens": 10_000_000},
    "openwebtext_100M": {"hf_path": "Skylion007/openwebtext", "split": "train", "max_tokens": 100_000_000},
    "fineweb_1B": {"hf_path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "max_tokens": 1_000_000_000},
    "fineweb_2_5B": {"hf_path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "max_tokens": 2_500_000_000},
}

SHARD_SIZE = 100_000_000

def tokenize_dataset(dataset_key: str, output_dir: str, val_ratio: float = 0.01):
    from datasets import load_dataset
    config = DATASET_CONFIGS[dataset_key]
    tokenizer = Tokenizer()
    out_path = Path(output_dir) / dataset_key
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {config['hf_path']}")
    load_kwargs = {"path": config["hf_path"], "split": config["split"], "streaming": True}
    if "name" in config:
        load_kwargs["name"] = config["name"]
    ds = load_dataset(**load_kwargs)

    max_tokens = config["max_tokens"]
    all_tokens = []
    total = 0

    print(f"Tokenizing up to {max_tokens:,} tokens...")
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer.encode(text)
        tokens.append(tokenizer.eot_token)
        all_tokens.extend(tokens)
        total += len(tokens)
        if total % 1_000_000 < len(tokens):
            print(f"  {total:,} tokens...")
        if total >= max_tokens:
            break

    all_tokens = all_tokens[:max_tokens]
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens):,}")

    val_size = int(len(all_tokens) * val_ratio)
    train_tokens = all_tokens[:-val_size] if val_size > 0 else all_tokens
    val_tokens = all_tokens[-val_size:] if val_size > 0 else np.array([], dtype=np.uint16)

    def write_shards(tokens, prefix):
        for i in range(0, len(tokens), SHARD_SIZE):
            shard = tokens[i : i + SHARD_SIZE]
            shard_path = out_path / f"{prefix}_{i // SHARD_SIZE:03d}.bin"
            shard.tofile(str(shard_path))
            print(f"  Wrote {shard_path} ({len(shard):,} tokens)")

    print("Writing train shards...")
    write_shards(train_tokens, "train")
    print("Writing val shards...")
    write_shards(val_tokens, "val")

    meta = {"vocab_size": tokenizer.vocab_size, "total_tokens": len(all_tokens), "train_tokens": len(train_tokens), "val_tokens": len(val_tokens), "val_ratio": val_ratio}
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Output: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare tokenized dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output", type=str, default="datasets/tokenized")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    args = parser.parse_args()
    tokenize_dataset(args.dataset, args.output, args.val_ratio)

if __name__ == "__main__":
    main()

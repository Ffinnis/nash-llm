"""Download and encode a dataset into binary shards for pretraining."""
import argparse
import json
import os
import numpy as np
import sys
from pathlib import Path
from nash_llm.data.tokenizer import Tokenizer

DATASET_CONFIGS = {
    "tinystories_10M": {"hf_path": "roneneldan/TinyStories", "split": "train", "max_tokens": 40_000_000},
    "openwebtext_100M": {"hf_path": "Skylion007/openwebtext", "split": "train", "max_tokens": 440_555_992},
    "fineweb_1B": {"hf_path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "max_tokens": 4_000_000_000},
    "fineweb_2_5B": {"hf_path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "max_tokens": 10_000_000_000},
}

SHARD_SIZE = 100_000_000


def resolve_token_dtype(tokenizer: Tokenizer, requested_dtype: str) -> np.dtype:
    if requested_dtype != "auto":
        dtype = np.dtype(requested_dtype)
        if tokenizer.vocab_size - 1 > np.iinfo(dtype).max:
            raise ValueError(
                f"{requested_dtype} cannot store vocab_size={tokenizer.vocab_size}; "
                "use a wider token dtype."
            )
        return dtype
    if tokenizer.vocab_size <= np.iinfo(np.uint8).max + 1:
        return np.dtype(np.uint8)
    if tokenizer.vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def load_streaming_dataset(dataset_key: str):
    from datasets import load_dataset

    config = DATASET_CONFIGS[dataset_key]
    load_kwargs = {"path": config["hf_path"], "split": config["split"], "streaming": True}
    if "name" in config:
        load_kwargs["name"] = config["name"]
    return load_dataset(**load_kwargs)


def calibrate_dataset(dataset_key: str, max_docs: int, max_gpt_tokens: int):
    byte_tokenizer = Tokenizer(representation="bytes")
    gpt_tokenizer = Tokenizer(representation="tiktoken", encoding_name="gpt2")
    ds = load_streaming_dataset(dataset_key)

    docs = 0
    total_bytes = 0
    total_gpt_tokens = 0
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        docs += 1
        total_bytes += len(byte_tokenizer.encode(text)) + 1
        total_gpt_tokens += len(gpt_tokenizer.encode(text)) + 1
        if docs >= max_docs or total_gpt_tokens >= max_gpt_tokens:
            break

    if total_gpt_tokens == 0:
        raise RuntimeError(f"No text examples found while calibrating {dataset_key}")

    bytes_per_gpt_token = total_bytes / total_gpt_tokens
    print(f"dataset: {dataset_key}")
    print(f"docs: {docs:,}")
    print(f"bytes: {total_bytes:,}")
    print(f"gpt2_tokens: {total_gpt_tokens:,}")
    print(f"bytes_per_gpt2_token: {bytes_per_gpt_token:.4f}")
    for token_budget in (10_000_000, 100_000_000, 1_000_000_000, 2_500_000_000):
        byte_budget = round(token_budget * bytes_per_gpt_token)
        print(f"{token_budget:,} gpt2-token-equivalent ~= {byte_budget:,} bytes")


def tokenize_dataset(
    dataset_key: str,
    output_dir: str,
    val_ratio: float = 0.01,
    representation: str = "bytes",
    tokenizer_encoding: str = "gpt2",
    token_dtype: str = "auto",
):
    config = DATASET_CONFIGS[dataset_key]
    tokenizer = Tokenizer(representation=representation, encoding_name=tokenizer_encoding)
    dtype = resolve_token_dtype(tokenizer, token_dtype)
    out_path = Path(output_dir) / dataset_key
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {config['hf_path']}")
    ds = load_streaming_dataset(dataset_key)

    max_tokens = config["max_tokens"]
    all_tokens = []
    total = 0

    unit_name = "bytes" if representation == "bytes" else "tokens"
    print(f"Encoding up to {max_tokens:,} {unit_name} with {representation}...")
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer.encode(text)
        tokens.append(tokenizer.eot_token)
        all_tokens.extend(tokens)
        total += len(tokens)
        if total % 1_000_000 < len(tokens):
            print(f"  {total:,} {unit_name}...")
        if total >= max_tokens:
            break

    all_tokens = all_tokens[:max_tokens]
    all_tokens = np.array(all_tokens, dtype=dtype)
    print(f"Total {unit_name}: {len(all_tokens):,}")

    val_size = int(len(all_tokens) * val_ratio)
    train_tokens = all_tokens[:-val_size] if val_size > 0 else all_tokens
    val_tokens = all_tokens[-val_size:] if val_size > 0 else np.array([], dtype=dtype)

    def write_shards(tokens, prefix):
        for i in range(0, len(tokens), SHARD_SIZE):
            shard = tokens[i : i + SHARD_SIZE]
            shard_path = out_path / f"{prefix}_{i // SHARD_SIZE:03d}.bin"
            shard.tofile(str(shard_path))
            print(f"  Wrote {shard_path} ({len(shard):,} {unit_name})")

    print("Writing train shards...")
    write_shards(train_tokens, "train")
    print("Writing val shards...")
    write_shards(val_tokens, "val")

    meta = {
        **tokenizer.metadata(),
        "token_dtype": dtype.name,
        "total_tokens": len(all_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "val_ratio": val_ratio,
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Output: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare encoded pretraining dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output", type=str, default="datasets/bytes")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--representation", type=str, default="bytes", choices=["bytes", "tiktoken"])
    parser.add_argument("--tokenizer_encoding", type=str, default="gpt2")
    parser.add_argument("--token_dtype", type=str, default="auto", choices=["auto", "uint8", "uint16", "uint32"])
    parser.add_argument("--calibrate", action="store_true", help="Measure bytes per GPT-2 token and exit")
    parser.add_argument("--calibrate_docs", type=int, default=10_000)
    parser.add_argument("--calibrate_gpt_tokens", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.calibrate:
        calibrate_dataset(args.dataset, args.calibrate_docs, args.calibrate_gpt_tokens)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    tokenize_dataset(
        args.dataset,
        args.output,
        args.val_ratio,
        representation=args.representation,
        tokenizer_encoding=args.tokenizer_encoding,
        token_dtype=args.token_dtype,
    )

if __name__ == "__main__":
    main()

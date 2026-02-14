"""Main training entry point with config + CLI overrides."""
import argparse
from nash_llm.config import load_config
from nash_llm.training import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Nash-LLM")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1
    return args, overrides

def main():
    args, overrides = parse_args()
    config = load_config(args.config, overrides=overrides if overrides else None)
    print(f"Model: {config.model}")
    print(f"Train: {config.train}")
    print(f"Data:  {config.data}")
    trainer = Trainer(config, checkpoint_dir=args.checkpoint_dir, resume_from=args.resume)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Total parameters: {total_params:,}")
    history = trainer.train()
    print(f"Training complete. {len(history)} steps.")

if __name__ == "__main__":
    main()

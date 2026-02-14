"""Generate text from a trained checkpoint."""
import argparse
import torch
from nash_llm.config import ModelConfig
from nash_llm.model import GPT
from nash_llm.training.checkpoint import load_checkpoint
from nash_llm.eval import generate_text

def main():
    parser = argparse.ArgumentParser(description="Generate text")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_ckpt = torch.load(args.checkpoint, weights_only=False)
    model_cfg = ModelConfig(**raw_ckpt["config"]["model"])
    model = GPT(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    text = generate_text(model, prompt=args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    print(text)

if __name__ == "__main__":
    main()

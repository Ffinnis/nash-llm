"""Plot training metrics from a wandb run."""
import argparse
import matplotlib.pyplot as plt
import wandb

def main():
    parser = argparse.ArgumentParser(description="Plot metrics from wandb run")
    parser.add_argument("--run_id", type=str, required=True, help="wandb run ID")
    parser.add_argument("--project", type=str, default="nash-llm")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="val_loss", help="Comma-separated metric names")
    parser.add_argument("--output", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    api = wandb.Api()
    run_path = f"{args.entity + '/' if args.entity else ''}{args.project}/{args.run_id}"
    run = api.run(run_path)
    history = run.scan_history()

    metrics = [m.strip() for m in args.metrics.split(",")]
    data = {m: {"steps": [], "values": []} for m in metrics}

    for row in history:
        step = row.get("_step", 0)
        for m in metrics:
            if m in row and row[m] is not None:
                data[m]["steps"].append(step)
                data[m]["values"].append(row[m])

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), squeeze=False)
    for i, m in enumerate(metrics):
        ax = axes[i][0]
        ax.plot(data[m]["steps"], data[m]["values"])
        ax.set_xlabel("Step")
        ax.set_ylabel(m)
        ax.set_title(f"{m} — {run.name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

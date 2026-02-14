"""Compare multiple wandb training runs."""
import argparse
import matplotlib.pyplot as plt
import wandb

def main():
    parser = argparse.ArgumentParser(description="Compare wandb runs")
    parser.add_argument("--runs", type=str, required=True, help="Comma-separated run IDs")
    parser.add_argument("--project", type=str, default="nash-llm")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="val_loss", help="Comma-separated metric names")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    api = wandb.Api()
    run_ids = [r.strip() for r in args.runs.split(",")]
    metrics = [m.strip() for m in args.metrics.split(",")]

    runs_data = {}
    configs = {}

    for run_id in run_ids:
        run_path = f"{args.entity + '/' if args.entity else ''}{args.project}/{run_id}"
        run = api.run(run_path)
        runs_data[run_id] = {"name": run.name, "metrics": {m: {"steps": [], "values": []} for m in metrics}}
        configs[run_id] = run.config
        for row in run.scan_history():
            step = row.get("_step", 0)
            for m in metrics:
                if m in row and row[m] is not None:
                    runs_data[run_id]["metrics"][m]["steps"].append(step)
                    runs_data[run_id]["metrics"][m]["values"].append(row[m])

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), squeeze=False)
    for i, m in enumerate(metrics):
        ax = axes[i][0]
        for run_id in run_ids:
            d = runs_data[run_id]["metrics"][m]
            label = runs_data[run_id]["name"] or run_id
            ax.plot(d["steps"], d["values"], label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel(m)
        ax.set_title(m)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()

    print(f"\n{'Run':<30} ", end="")
    for m in metrics:
        print(f"{m:<15} ", end="")
    print()
    print("-" * (30 + 16 * len(metrics)))
    for run_id in run_ids:
        name = runs_data[run_id]["name"] or run_id
        print(f"{name:<30} ", end="")
        for m in metrics:
            vals = runs_data[run_id]["metrics"][m]["values"]
            final = f"{vals[-1]:.4f}" if vals else "N/A"
            print(f"{final:<15} ", end="")
        print()

    if len(run_ids) == 2:
        print(f"\nConfig diff between {run_ids[0]} and {run_ids[1]}:")
        c1, c2 = configs[run_ids[0]], configs[run_ids[1]]
        all_keys = set(list(c1.keys()) + list(c2.keys()))
        for key in sorted(all_keys):
            v1 = c1.get(key)
            v2 = c2.get(key)
            if v1 != v2:
                print(f"  {key}: {v1} -> {v2}")

if __name__ == "__main__":
    main()

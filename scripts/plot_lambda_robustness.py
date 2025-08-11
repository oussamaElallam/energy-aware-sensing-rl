import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_aggregates(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    aggr = data.get("aggregates", {})
    # keys may be strings like "0.0", "20.0"
    items = []
    for k, v in aggr.items():
        try:
            lam = float(k)
        except Exception:
            continue
        items.append((lam, v))
    items.sort(key=lambda x: x[0])
    lambdas = [lam for lam, _ in items]
    det_mean = [it.get("mean_detection", None) for _, it in items]
    det_std = [it.get("std_detection", None) for _, it in items]
    eng_mean = [it.get("mean_mAh", None) for _, it in items]
    eng_std = [it.get("std_mAh", None) for _, it in items]
    return lambdas, det_mean, det_std, eng_mean, eng_std


def plot_summary(lambdas, det_mean, det_std, eng_mean, eng_std, out_path: Path):
    plt.rcParams.update({
        "font.size": 11,
        "figure.figsize": (9.5, 3.2),
        "axes.grid": True,
    })

    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.1]})

    # 1) Detection vs lambda (mean ± std)
    ax = axes[0]
    ax.errorbar(lambdas, det_mean, yerr=det_std, fmt="o-", capsize=4, lw=1.5, color="#1f77b4")
    ax.set_xlabel("λ")
    ax.set_ylabel("Detection (%)")
    ax.set_title("Detection vs λ")

    # 2) Energy vs lambda (mean ± std)
    ax = axes[1]
    ax.errorbar(lambdas, eng_mean, yerr=eng_std, fmt="s-", capsize=4, lw=1.5, color="#d62728")
    ax.set_xlabel("λ")
    ax.set_ylabel("Energy (mAh / 16h)")
    ax.set_title("Energy vs λ")

    # 3) Detection vs Energy with annotations and path
    ax = axes[2]
    ax.scatter(eng_mean, det_mean, s=40, color="#2ca02c", zorder=3)
    # connect in order of lambda
    ax.plot(eng_mean, det_mean, "-", color="#888888", alpha=0.7, zorder=2)
    # annotate with lambda values
    for x, y, lam in zip(eng_mean, det_mean, lambdas):
        ax.annotate(f"λ={int(lam) if lam.is_integer() else lam}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Energy (mAh / 16h)")
    ax.set_ylabel("Detection (%)")
    ax.set_title("Detection vs Energy")

    fig.suptitle("Policy Robustness across λ (mean ± std)")
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot lambda robustness summary from results JSON")
    p.add_argument("--json", type=str, default="paper_results/paper_results_lambdaA.json")
    p.add_argument("--out", type=str, default="paper_results/lambda_robustness_summary.png")
    args = p.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    lambdas, det_mean, det_std, eng_mean, eng_std = load_aggregates(json_path)
    if not lambdas:
        raise ValueError("No lambda aggregates found in JSON")

    plot_summary(lambdas, det_mean, det_std, eng_mean, eng_std, Path(args.out))


if __name__ == "__main__":
    main()

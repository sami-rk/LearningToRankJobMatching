import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_metric_comparison(results_df: pd.DataFrame) -> str:
    """Bar chart comparing NDCG@10 and MAP@5 across all models."""
    ensure_output_dir()

    models = results_df["model"].values
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2,
        results_df["ndcg10_mean"],
        width,
        yerr=results_df["ndcg10_std"],
        label="NDCG@10",
        color="#4C72B0",
        capsize=5,
    )
    bars2 = ax.bar(
        x + width / 2,
        results_df["map5_mean"],
        width,
        yerr=results_df["map5_std"],
        label="MAP@5",
        color="#55A868",
        capsize=5,
    )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: NDCG@10 and MAP@5", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_fold_distribution(results_df: pd.DataFrame) -> str:
    """Box plot showing metric variance per model across folds."""
    ensure_output_dir()

    rows = []
    for _, row in results_df.iterrows():
        for i, ndcg in enumerate(row["fold_ndcgs"]):
            rows.append({"Model": row["model"], "Metric": "NDCG@10", "Fold": i + 1, "Value": ndcg})
        for i, mp in enumerate(row["fold_maps"]):
            rows.append({"Model": row["model"], "Metric": "MAP@5", "Fold": i + 1, "Value": mp})

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df[df["Metric"] == "NDCG@10"], x="Model", y="Value", hue="Model", ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("NDCG@10 Fold Distribution", fontsize=13)
    axes[0].set_ylabel("NDCG@10", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    sns.boxplot(data=df[df["Metric"] == "MAP@5"], x="Model", y="Value", hue="Model", ax=axes[1], palette="Set2", legend=False)
    axes[1].set_title("MAP@5 Fold Distribution", fontsize=13)
    axes[1].set_ylabel("MAP@5", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fold_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_radar_chart(results_df: pd.DataFrame) -> str:
    """Radar/spider chart comparing models across metrics."""
    ensure_output_dir()

    categories = ["NDCG@10", "MAP@5"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for idx, (_, row) in enumerate(results_df.iterrows()):
        values = [row["ndcg10_mean"], row["map5_mean"]]
        values += values[:1]
        color = colors[idx % len(colors)]

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=row["model"], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Multi-Metric Model Comparison", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "radar_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(results_df: pd.DataFrame) -> list[str]:
    """Generate all comparison plots and return file paths."""
    paths = []
    paths.append(plot_metric_comparison(results_df))
    paths.append(plot_fold_distribution(results_df))
    paths.append(plot_radar_chart(results_df))
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    for p in paths:
        print(f"  -> {os.path.basename(p)}")
    return paths

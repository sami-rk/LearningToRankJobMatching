"""Hyperparameter finetuning tool — random search with comparison plots.

Usage (from project root):
    python -m finetune --trials 20              # all models, 20 trials each
    python -m finetune --models XGBoost LightGBM --trials 10
    python -m finetune --trials 5 --seed 123
"""

import argparse
import json
import os
import sys
import time
from itertools import product

# Allow running from inside the project directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from LearningToRankJobMatching.config import (
    FEATURE_COLS,
    JOB_COLS,
    CAND_COLS,
    MODEL_PARAMS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    CATBOOST_PARAMS,
    N_SPLITS,
    SEED,
)
from LearningToRankJobMatching.data.loader import load_datasets
from LearningToRankJobMatching.preprocessing.jobs import preprocess_jobs
from LearningToRankJobMatching.preprocessing.candidates import preprocess_candidates
from LearningToRankJobMatching.features.interactions import build_features
from LearningToRankJobMatching.features.encoding import oof_target_encode
from LearningToRankJobMatching.models.ranker import create_ranker, cross_validate_model
from LearningToRankJobMatching.evaluation.metrics import compute_ndcg10, compute_map5

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "finetune_output")

# ── Parameter search spaces ──────────────────────────────────────────────────
# Each key maps to (type, low, high) where type is "int" or "float".
# float ranges use log-uniform sampling for learning rates, uniform for subsample.

PARAM_SPACES = {
    "GradientBoosting": {
        "n_estimators": ("int", 100, 500),
        "learning_rate": ("log_float", 0.01, 0.2),
        "max_depth": ("int", 3, 8),
        "min_samples_leaf": ("int", 5, 50),
        "subsample": ("float", 0.6, 1.0),
    },
    "XGBoost": {
        "n_estimators": ("int", 100, 500),
        "learning_rate": ("log_float", 0.01, 0.2),
        "max_depth": ("int", 3, 8),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "LightGBM": {
        "n_estimators": ("int", 100, 500),
        "learning_rate": ("log_float", 0.01, 0.2),
        "num_leaves": ("int", 15, 63),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "CatBoost": {
        "iterations": ("int", 100, 500),
        "learning_rate": ("log_float", 0.01, 0.2),
        "depth": ("int", 3, 8),
    },
}

# Fixed params that should NOT be tuned (passed through as-is)
FIXED_PARAMS = {
    "GradientBoosting": {"random_state": SEED},
    "XGBoost": {"objective": "rank:ndcg", "random_state": SEED, "verbosity": 0},
    "LightGBM": {"objective": "lambdarank", "random_state": SEED, "verbose": -1},
    "CatBoost": {"loss_function": "YetiRank", "random_seed": SEED, "verbose": 0},
}

DEFAULT_PARAMS = {
    "GradientBoosting": MODEL_PARAMS,
    "XGBoost": XGBOOST_PARAMS,
    "LightGBM": LIGHTGBM_PARAMS,
    "CatBoost": CATBOOST_PARAMS,
}


def _sample_param(rng, spec):
    """Sample a single parameter from its (type, low, high) spec."""
    kind, lo, hi = spec
    if kind == "int":
        return int(rng.integers(lo, hi + 1))
    elif kind == "float":
        return round(rng.uniform(lo, hi), 4)
    elif kind == "log_float":
        return round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 4)
    raise ValueError(f"Unknown param type: {kind}")


def sample_params(model_name, rng):
    """Sample a full param dict for a model, merging with fixed params."""
    space = PARAM_SPACES[model_name]
    sampled = {k: _sample_param(rng, v) for k, v in space.items()}
    sampled.update(FIXED_PARAMS[model_name])
    return sampled


def load_data():
    """Load and preprocess data, return (X_train, y_train, groups)."""
    jobs, candidates, train, test = load_datasets()
    jobs = preprocess_jobs(jobs)
    candidates = preprocess_candidates(candidates)
    train_feat = build_features(train, jobs, candidates)
    test_feat = build_features(test, jobs, candidates)
    train_feat["industry_enc"], test_feat["industry_enc"] = oof_target_encode(
        train_feat, test_feat, column="industry"
    )
    X_train = train_feat[FEATURE_COLS].fillna(0).values
    y_train = train_feat["relevance_label"].values
    groups = train_feat["job_id"].values
    print(f"Data loaded: X={X_train.shape}, y={y_train.shape}\n")
    return X_train, y_train, groups


def run_baseline(model_name, X, y, groups):
    """Cross-validate with default params, return result dict."""
    params = dict(DEFAULT_PARAMS[model_name])
    params.update(FIXED_PARAMS[model_name])
    return cross_validate_model(model_name, X, y, groups, params)


def run_trial(model_name, params, X, y, groups):
    """Cross-validate with sampled params, return result dict."""
    return cross_validate_model(model_name, X, y, groups, params)


def run_finetune(models, n_trials, seed):
    """Main finetuning loop."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    X, y, groups = load_data()

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Finetuning: {model_name}")
        print(f"{'='*60}")

        # Baseline
        print(f"\n  [Baseline] Default params:")
        baseline = run_baseline(model_name, X, y, groups)
        baseline["trial"] = "baseline"
        baseline["params"] = dict(DEFAULT_PARAMS[model_name])
        all_results.append(baseline)
        print(f"    NDCG@10: {baseline['ndcg10_mean']:.4f} | MAP@5: {baseline['map5_mean']:.4f}")

        # Random search
        best_ndcg = baseline["ndcg10_mean"]
        best_params = None

        for trial in tqdm(range(1, n_trials + 1), desc=f"  {model_name} trials", unit="trial"):
            params = sample_params(model_name, rng)

            # Strip fixed params for display
            display_params = {k: v for k, v in params.items() if k not in FIXED_PARAMS[model_name]}

            result = run_trial(model_name, params, X, y, groups)
            result["trial"] = trial
            result["params"] = display_params
            all_results.append(result)

            if result["ndcg10_mean"] > best_ndcg:
                best_ndcg = result["ndcg10_mean"]
                best_params = display_params
                tqdm.write(
                    f"    Trial {trial}: NDCG@10={result['ndcg10_mean']:.4f} *NEW BEST* "
                    f"({display_params})"
                )
            else:
                tqdm.write(
                    f"    Trial {trial}: NDCG@10={result['ndcg10_mean']:.4f} "
                    f"(best={best_ndcg:.4f})"
                )

        # Summary
        improvement = best_ndcg - baseline["ndcg10_mean"]
        print(f"\n  Summary for {model_name}:")
        print(f"    Baseline NDCG@10:  {baseline['ndcg10_mean']:.4f}")
        print(f"    Best tuned NDCG@10: {best_ndcg:.4f}")
        print(f"    Improvement:        {improvement:+.4f}")
        if best_params:
            print(f"    Best params:        {best_params}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "finetune_results.json")
    serializable = []
    for r in all_results:
        entry = {
            "model": r["model"],
            "trial": r["trial"],
            "ndcg10_mean": r["ndcg10_mean"],
            "ndcg10_std": r["ndcg10_std"],
            "map5_mean": r["map5_mean"],
            "map5_std": r["map5_std"],
            "params": r["params"],
        }
        serializable.append(entry)
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    generate_finetune_plots(all_results)

    return all_results


def generate_finetune_plots(all_results):
    """Generate comparison plots for finetuning results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = list(dict.fromkeys(r["model"] for r in all_results))

    # ── Plot 1: Baseline vs Best per model (grouped bar) ──
    rows = []
    for model_name in models:
        model_results = [r for r in all_results if r["model"] == model_name]
        baseline = [r for r in model_results if r["trial"] == "baseline"][0]
        trials = [r for r in model_results if r["trial"] != "baseline"]
        best = max(trials, key=lambda r: r["ndcg10_mean"]) if trials else baseline

        rows.append({"Model": model_name, "Config": "Baseline", "NDCG@10": baseline["ndcg10_mean"], "MAP@5": baseline["map5_mean"]})
        rows.append({"Model": model_name, "Config": "Best Tuned", "NDCG@10": best["ndcg10_mean"], "MAP@5": best["map5_mean"]})

    df_bar = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric in zip(axes, ["NDCG@10", "MAP@5"]):
        sns.barplot(data=df_bar, x="Model", y=metric, hue="Config", ax=ax, palette=["#4C72B0", "#55A868"])
        ax.set_title(f"{metric}: Baseline vs Best Tuned", fontsize=13)
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", fontsize=9, padding=3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "baseline_vs_tuned.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Plot 2: Trial-by-trial NDCG@10 (line plot per model) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for idx, model_name in enumerate(models):
        model_results = [r for r in all_results if r["model"] == model_name]
        baseline = [r for r in model_results if r["trial"] == "baseline"][0]
        trials = sorted([r for r in model_results if r["trial"] != "baseline"], key=lambda r: r["trial"])

        trial_nums = [0] + [r["trial"] for r in trials]
        ndcgs = [baseline["ndcg10_mean"]] + [r["ndcg10_mean"] for r in trials]

        ax.plot(trial_nums, ndcgs, marker="o", markersize=4, label=model_name, color=colors[idx % len(colors)], alpha=0.8)

    ax.axhline(y=max(r["ndcg10_mean"] for r in all_results if r["trial"] != "baseline"), color="gray", linestyle="--", alpha=0.5, label="Global best")
    ax.set_xlabel("Trial #", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_title("Finetuning Progress: NDCG@10 per Trial", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "finetuning_progress.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Plot 3: Parameter sensitivity (scatter: learning_rate vs NDCG) ──
    param_rows = []
    for r in all_results:
        if r["trial"] == "baseline":
            continue
        p = r["params"]
        for param_name in ["learning_rate", "n_estimators", "max_depth", "num_leaves", "depth", "subsample", "colsample_bytree"]:
            if param_name in p:
                param_rows.append({
                    "Model": r["model"],
                    "Param": param_name,
                    "Value": p[param_name],
                    "NDCG@10": r["ndcg10_mean"],
                })

    if param_rows:
        df_params = pd.DataFrame(param_rows)
        g = sns.FacetGrid(df_params, col="Param", col_wrap=3, sharex=False, height=4, aspect=1.2)
        g.map_dataframe(sns.scatterplot, x="Value", y="NDCG@10", hue="Model", alpha=0.7, s=40)
        g.add_legend()
        g.set_titles("{col_name}")
        g.set_axis_labels("Value", "NDCG@10")

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "param_sensitivity.png")
        g.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # ── Plot 4: Best config summary table ──
    summary_rows = []
    for model_name in models:
        model_results = [r for r in all_results if r["model"] == model_name]
        baseline = [r for r in model_results if r["trial"] == "baseline"][0]
        trials = [r for r in model_results if r["trial"] != "baseline"]
        best = max(trials, key=lambda r: r["ndcg10_mean"]) if trials else baseline
        improvement = best["ndcg10_mean"] - baseline["ndcg10_mean"]

        summary_rows.append({
            "Model": model_name,
            "Baseline NDCG@10": f"{baseline['ndcg10_mean']:.4f}",
            "Best NDCG@10": f"{best['ndcg10_mean']:.4f}",
            "Improvement": f"{improvement:+.4f}",
            "Best Config": str(best["params"]),
        })

    df_summary = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(14, 2 + len(summary_rows) * 0.6))
    ax.axis("off")
    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title("Finetuning Summary", fontsize=14, pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "finetune_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter finetuning via random search")
    parser.add_argument(
        "--models", nargs="+",
        choices=["GradientBoosting", "XGBoost", "LightGBM", "CatBoost"],
        default=["GradientBoosting", "XGBoost", "LightGBM", "CatBoost"],
        help="Models to finetune (default: all four)",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of random search trials per model (default: 20)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_finetune(args.models, args.trials, args.seed)


if __name__ == "__main__":
    main()

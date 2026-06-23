from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np

from task3.config import FEATURE_COLS, MODEL_PARAMS, N_SPLITS
from task3.evaluation.metrics import compute_ndcg10, compute_map5


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float]:
    """Run GroupKFold CV and return (mean_ndcg10, mean_map5)."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_ndcgs, oof_maps = [], []

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(X, y, groups)
    ):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        g_val = groups[val_idx]

        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X_tr, y_tr)

        val_df = pd.DataFrame(
            {"job_id": g_val, "score": model.predict(X_val), "rel": y_val}
        )

        fold_ndcg = compute_ndcg10(val_df)
        fold_map5 = compute_map5(val_df)

        oof_ndcgs.append(fold_ndcg)
        oof_maps.append(fold_map5)
        print(
            f"  Fold {fold + 1} | NDCG@10: {fold_ndcg:.4f} | MAP@5: {fold_map5:.4f}"
        )

    mean_ndcg = float(np.mean(oof_ndcgs))
    std_ndcg = float(np.std(oof_ndcgs))
    mean_map = float(np.mean(oof_maps))
    std_map = float(np.std(oof_maps))

    print(f"\nMean OOF NDCG@10:\t{mean_ndcg:.4f} ± {std_ndcg:.4f}")
    print(f"Mean OOF MAP@5:\t\t{mean_map:.4f} ± {std_map:.4f}")

    return mean_ndcg, mean_map


def train_final_model(
    X: np.ndarray, y: np.ndarray
) -> GradientBoostingRegressor:
    """Train a GradientBoostingRegressor on the full training data."""
    model = GradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    """Return continuous relevance scores."""
    return model.predict(X)

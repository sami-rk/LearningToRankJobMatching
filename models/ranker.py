import os
from abc import ABC, abstractmethod
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

from LearningToRankJobMatching.config import (
    FEATURE_COLS,
    MODEL_PARAMS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    CATBOOST_PARAMS,
    N_SPLITS,
    MODEL_CACHE_DIR,
)
from LearningToRankJobMatching.evaluation.metrics import compute_ndcg10, compute_map5


def _sort_by_groups(X, y, groups):
    """Sort X, y, groups so that group IDs are in non-decreasing order.

    Required by XGBoost and LightGBM ranking objectives.
    """
    order = np.argsort(groups, kind="mergesort")
    return X[order], y[order], groups[order]


class BaseRanker(ABC):
    """Abstract base class for all rankers."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


class GradientBoostingRanker(BaseRanker):
    def __init__(self, params: dict = None):
        self.params = params or MODEL_PARAMS
        self.model = GradientBoostingRegressor(**self.params)

    def fit(self, X, y, groups=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGBoostRanker(BaseRanker):
    def __init__(self, params: dict = None):
        self.params = params or XGBOOST_PARAMS
        self.model = None

    def fit(self, X, y, groups=None):
        import xgboost as xgb

        self.model = xgb.XGBRanker(**self.params)
        if groups is not None:
            X_s, y_s, g_s = _sort_by_groups(X, y, groups)
            _, qids = np.unique(g_s, return_inverse=True)
            self.model.fit(X_s, y_s, qid=qids)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LightGBMRanker(BaseRanker):
    def __init__(self, params: dict = None):
        self.params = params or LIGHTGBM_PARAMS
        self.model = None

    def fit(self, X, y, groups=None):
        import lightgbm as lgb

        self.model = lgb.LGBMRanker(**self.params)
        if groups is not None:
            X_s, y_s, g_s = _sort_by_groups(X, y, groups)
            _, indices = np.unique(g_s, return_inverse=True)
            qids = np.bincount(indices)
            self.model.fit(X_s, y_s, group=qids)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class CatBoostRanker(BaseRanker):
    def __init__(self, params: dict = None):
        self.params = params or CATBOOST_PARAMS
        self.model = None

    def fit(self, X, y, groups=None):
        from catboost import CatBoost, Pool

        self.model = CatBoost(self.params)
        if groups is not None:
            qids = self._groups_to_qids(groups)
            pool = Pool(X, label=y, group_id=qids)
            self.model.fit(pool)
        else:
            pool = Pool(X, label=y)
            self.model.fit(pool)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def _groups_to_qids(groups: np.ndarray) -> np.ndarray:
        _, indices = np.unique(groups, return_inverse=True)
        return indices


RANKER_CLASSES = {
    "GradientBoosting": GradientBoostingRanker,
    "XGBoost": XGBoostRanker,
    "LightGBM": LightGBMRanker,
    "CatBoost": CatBoostRanker,
}


def create_ranker(name: str, params: dict = None) -> BaseRanker:
    """Factory to create a ranker by name."""
    if name not in RANKER_CLASSES:
        raise ValueError(f"Unknown model: {name}. Choose from {list(RANKER_CLASSES)}")
    return RANKER_CLASSES[name](params)


# ── Model caching ────────────────────────────────────────────────────────────

def _cache_path(model_name: str) -> str:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return os.path.join(MODEL_CACHE_DIR, f"{model_name}.joblib")


def _cv_cache_path(model_name: str) -> str:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return os.path.join(MODEL_CACHE_DIR, f"{model_name}_cv_results.joblib")


def has_cached_model(model_name: str) -> bool:
    return os.path.exists(_cache_path(model_name))


def has_cached_cv(model_name: str) -> bool:
    return os.path.exists(_cv_cache_path(model_name))


def save_model(model: BaseRanker, model_name: str) -> None:
    joblib.dump(model, _cache_path(model_name))


def load_model(model_name: str) -> BaseRanker:
    return joblib.load(_cache_path(model_name))


def save_cv_results(results: dict, model_name: str) -> None:
    joblib.dump(results, _cv_cache_path(model_name))


def load_cv_results(model_name: str) -> dict:
    return joblib.load(_cv_cache_path(model_name))


def ask_retrain(model_name: str) -> bool:
    """Ask user whether to retrain or use cached model."""
    cache_p = _cache_path(model_name)
    cv_cache_p = _cv_cache_path(model_name)

    has_model = os.path.exists(cache_p)
    has_cv = os.path.exists(cv_cache_p)

    if has_cv:
        cached = load_cv_results(model_name)
        print(f"  Found cached CV results for {model_name}:")
        print(f"    NDCG@10: {cached['ndcg10_mean']:.4f} ± {cached['ndcg10_std']:.4f}")
        print(f"    MAP@5:   {cached['map5_mean']:.4f} ± {cached['map5_std']:.4f}")

    if has_model:
        print(f"  Found cached trained model: {os.path.basename(cache_p)}")

    choice = input(f"  Retrain {model_name}? [y/N]: ").strip().lower()
    return choice in ("y", "yes")


# ── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict = None,
) -> dict:
    """Run GroupKFold CV for a single model. Returns dict with fold and mean metrics."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_ndcgs, fold_maps = [], []

    folds = list(gkf.split(X, y, groups))

    for fold, (tr_idx, val_idx) in enumerate(tqdm(folds, desc=f"  {model_name}", unit="fold", leave=True)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        g_val = groups[val_idx]

        tqdm.write(f"  Fold {fold + 1}/{N_SPLITS} — training...")

        ranker = create_ranker(model_name, params)
        ranker.fit(X_tr, y_tr, groups=groups[tr_idx])

        val_df = pd.DataFrame(
            {"job_id": g_val, "score": ranker.predict(X_val), "rel": y_val}
        )

        fold_ndcg = compute_ndcg10(val_df)
        fold_map = compute_map5(val_df)

        fold_ndcgs.append(fold_ndcg)
        fold_maps.append(fold_map)
        tqdm.write(
            f"  Fold {fold + 1}/{N_SPLITS} — NDCG@10: {fold_ndcg:.4f} | MAP@5: {fold_map:.4f}"
        )

    mean_ndcg = float(np.mean(fold_ndcgs))
    std_ndcg = float(np.std(fold_ndcgs))
    mean_map = float(np.mean(fold_maps))
    std_map = float(np.std(fold_maps))

    return {
        "model": model_name,
        "ndcg10_mean": mean_ndcg,
        "ndcg10_std": std_ndcg,
        "map5_mean": mean_map,
        "map5_std": std_map,
        "fold_ndcgs": fold_ndcgs,
        "fold_maps": fold_maps,
    }


def cross_validate_all(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Cross-validate all registered models and return results DataFrame."""
    from LearningToRankJobMatching.config import MODEL_REGISTRY

    results = []
    model_names = list(MODEL_REGISTRY.keys())

    for model_name in tqdm(model_names, desc="Models", unit="model"):

        if has_cached_cv(model_name) and not ask_retrain(model_name):
            result = load_cv_results(model_name)
            print(f"\n  Loaded cached CV for {model_name}")
        else:
            print(f"\n{'='*50}")
            print(f"  Training: {model_name}")
            print(f"{'='*50}")
            _, params = MODEL_REGISTRY[model_name]
            result = cross_validate_model(model_name, X, y, groups, params)
            save_cv_results(result, model_name)
            print(
                f"\n  Mean OOF NDCG@10:\t{result['ndcg10_mean']:.4f} ± {result['ndcg10_std']:.4f}"
            )
            print(
                f"  Mean OOF MAP@5:\t{result['map5_mean']:.4f} ± {result['map5_std']:.4f}"
            )

        results.append(result)

    return pd.DataFrame(results)


def train_final_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray = None,
) -> BaseRanker:
    """Train the best model on full training data. Checks cache first."""
    from LearningToRankJobMatching.config import MODEL_REGISTRY

    if has_cached_model(model_name) and not ask_retrain(model_name):
        print(f"  Loading cached {model_name} model...")
        return load_model(model_name)

    _, params = MODEL_REGISTRY[model_name]
    ranker = create_ranker(model_name, params)

    print(f"\n  Training final {model_name} on full data...")
    ranker.fit(X, y, groups)
    print(f"  Training complete.")

    save_model(ranker, model_name)
    print(f"  Model saved to {_cache_path(model_name)}")
    return ranker


def predict(model: BaseRanker, X: np.ndarray) -> np.ndarray:
    """Return continuous relevance scores."""
    return model.predict(X)

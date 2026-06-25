"""Main pipeline: load -> preprocess -> feature-engineer -> train -> submit."""

import numpy as np
import pandas as pd

from LearningToRankJobMatching.config import (
    FEATURE_COLS,
    JOB_COLS,
    CAND_COLS,
    SUBMISSION_PATH,
)
from LearningToRankJobMatching.data.loader import load_datasets
from LearningToRankJobMatching.preprocessing.jobs import preprocess_jobs
from LearningToRankJobMatching.preprocessing.candidates import preprocess_candidates
from LearningToRankJobMatching.features.interactions import build_features
from LearningToRankJobMatching.features.encoding import oof_target_encode
from LearningToRankJobMatching.models.ranker import cross_validate_all, train_final_model, predict
from LearningToRankJobMatching.evaluation.visuals import generate_all_plots


def run() -> None:
    """End-to-end pipeline."""

    # 1 -- Load ----------------------------------------------------------------
    jobs, candidates, train, test = load_datasets()

    # 2 -- Preprocess ----------------------------------------------------------
    jobs = preprocess_jobs(jobs)
    candidates = preprocess_candidates(candidates)

    # 3 -- Feature engineering --------------------------------------------------
    train_feat = build_features(train, jobs, candidates)
    test_feat = build_features(test, jobs, candidates)

    # 4 -- OOF target encoding for industry -------------------------------------
    train_feat["industry_enc"], test_feat["industry_enc"] = oof_target_encode(
        train_feat, test_feat, column="industry"
    )

    # 5 -- Prepare arrays ------------------------------------------------------
    X_train = train_feat[FEATURE_COLS].fillna(0).values
    y_train = train_feat["relevance_label"].values
    groups = train_feat["job_id"].values
    X_test = test_feat[FEATURE_COLS].fillna(0).values

    print(f"\nX_train: {X_train.shape} | X_test: {X_test.shape}\n")

    # 6 -- Cross-validate all models -------------------------------------------
    results = cross_validate_all(X_train, y_train, groups)

    # 7 -- Generate comparison plots --------------------------------------------
    generate_all_plots(results)

    # 8 -- Find best model and train on full data -------------------------------
    best_idx = results["ndcg10_mean"].idxmax()
    best_model_name = results.loc[best_idx, "model"]
    print(f"\nBest model: {best_model_name} (NDCG@10: {results.loc[best_idx, 'ndcg10_mean']:.4f})")

    model = train_final_model(best_model_name, X_train, y_train, groups)

    # 9 -- Predict & save ------------------------------------------------------
    test_feat["score"] = predict(model, X_test)

    submission = test_feat[["application_id", "score"]].sort_values("application_id")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved -> {SUBMISSION_PATH}")


if __name__ == "__main__":
    run()

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from task3.config import N_SPLITS


def oof_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column: str,
    target: str = "relevance_label",
    n_splits: int = N_SPLITS,
) -> tuple[pd.Series, pd.Series]:
    """Out-of-fold target encoding for *column* on train, full-data on test.

    Returns (train_encoded, test_encoded) as new Series.
    """
    global_mean = train_df[target].mean()
    gkf = GroupKFold(n_splits=n_splits)

    train_encoded = pd.Series(global_mean, index=train_df.index, name=f"{column}_enc")

    for tr_idx, val_idx in gkf.split(
        train_df, train_df[target], train_df["job_id"]
    ):
        means = train_df.iloc[tr_idx].groupby(column)[target].mean()
        train_encoded.iloc[val_idx] = (
            train_df.iloc[val_idx][column].map(means).fillna(global_mean)
        )

    full_means = train_df.groupby(column)[target].mean()
    test_encoded = (
        test_df[column].map(full_means).fillna(global_mean)
    )

    return train_encoded, test_encoded

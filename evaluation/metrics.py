import numpy as np
import pandas as pd


def compute_ndcg10(val_df: pd.DataFrame) -> float:
    """Mean NDCG@10 across all job groups in *val_df*.

    Expects columns: job_id, score, rel.
    """
    ndcgs: list[float] = []
    for _, grp in val_df.groupby("job_id"):
        grp = grp.sort_values("score", ascending=False)
        k = min(10, len(grp))

        rel_pred = grp["rel"].values[:k]
        dcg = sum(
            (2 ** rel_pred[i] - 1) / np.log2(i + 2) for i in range(k)
        )

        ideal_rels = sorted(grp["rel"].values, reverse=True)[:k]
        idcg = sum(
            (2 ** ideal_rels[i] - 1) / np.log2(i + 2) for i in range(k)
        )

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))


def compute_map5(val_df: pd.DataFrame) -> float:
    """Mean Average Precision@5 across all job groups in *val_df*.

    Relevant = label >= 3.
    Expects columns: job_id, score, rel.
    """
    aps: list[float] = []
    for _, grp in val_df.groupby("job_id"):
        total_relevant = int(np.sum(grp["rel"].values >= 3))
        if total_relevant == 0:
            aps.append(0.0)
            continue

        grp_top5 = grp.sort_values("score", ascending=False).head(5)
        rels_top5 = (grp_top5["rel"].values >= 3).astype(int)

        precisions: list[float] = []
        rel_count = 0
        for i, is_rel in enumerate(rels_top5):
            if is_rel == 1:
                rel_count += 1
                precisions.append(rel_count / (i + 1))

        ap5 = sum(precisions) / min(total_relevant, 5)
        aps.append(ap5)

    return float(np.mean(aps))

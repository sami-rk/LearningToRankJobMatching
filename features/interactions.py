import pandas as pd


def build_features(df: pd.DataFrame, jobs: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Merge job/candidate info and build interaction features.

    Returns a new DataFrame (never mutates the inputs).
    """
    from task3.config import JOB_COLS, CAND_COLS

    df = df.copy()
    df = df.merge(jobs[JOB_COLS], on="job_id", how="left")
    df = df.merge(candidates[CAND_COLS], on="candidate_id", how="left")

    # Ensure skill columns are safe sets after merge
    df["req_skills_set"] = df["req_skills_set"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    df["cand_skills_set"] = df["cand_skills_set"].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    # ── Feature 1: skill overlap (Jaccard-like) ───────────────────────────────
    df["skill_overlap"] = df.apply(
        lambda r: len(r["req_skills_set"] & r["cand_skills_set"])
        / max(len(r["req_skills_set"] | r["cand_skills_set"]), 1),
        axis=1,
    )

    # ── Feature 2: skill coverage (% of required skills matched) ───────────────
    df["skill_coverage"] = df.apply(
        lambda r: len(r["req_skills_set"] & r["cand_skills_set"])
        / max(len(r["req_skills_set"]), 1),
        axis=1,
    )
    df["num_matching_skills"] = df.apply(
        lambda r: len(r["req_skills_set"] & r["cand_skills_set"]), axis=1
    )
    df["req_skills_count"] = df["req_skills_set"].apply(len)

    # ── Feature 3: experience gap ──────────────────────────────────────────────
    df["exp_gap_min"] = (
        df["years_experience"] - df["min_years_experience"].fillna(0)
    ).fillna(0)
    df["exp_in_range"] = (
        (df["years_experience"] >= df["min_years_experience"].fillna(0))
        & (df["years_experience"] <= df["max_years_experience"].fillna(999))
    ).astype(int)

    # ── Feature 4: salary compatibility ────────────────────────────────────────
    df["salary_ratio"] = (
        df["expected_salary"] / (df["salary_mid_usd"] + 1)
    ).fillna(1)
    df["salary_affordable"] = (
        df["expected_salary"] <= df["salary_max_usd"]
    ).astype(int)

    # ── Feature 5: seniority compatibility ─────────────────────────────────────
    df["seniority_match"] = (df["cand_seniority"] == df["job_seniority"]).astype(int)
    df["seniority_diff"] = df["cand_seniority"] - df["job_seniority"]

    # ── Feature 6: location compatibility ──────────────────────────────────────
    df["location_compat"] = df.apply(_location_compat, axis=1)

    # ── Temporal features ──────────────────────────────────────────────────────
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df["job_posted_date"] = pd.to_datetime(
        df["job_posted_date"], errors="coerce", dayfirst=True
    )
    df["days_since_posted"] = (
        (df["application_date"] - df["job_posted_date"]).dt.days.fillna(0)
    )
    df["app_month"] = df["application_date"].dt.month.fillna(-1).astype(int)

    return df


def _location_compat(row: pd.Series) -> float:
    """Heuristic location compatibility score [0, 1]."""
    if row["remote_num"] == 2:
        return 1.0  # fully remote
    if row["willing_to_relocate_bin"] == 1:
        return 0.7  # willing to relocate
    jloc = str(row.get("job_location", "")).lower().strip()
    cloc = str(row.get("candidate_location", "")).lower().strip()
    if jloc and cloc and (jloc in cloc or cloc in jloc):
        return 1.0
    return 0.3

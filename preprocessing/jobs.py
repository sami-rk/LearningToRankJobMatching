import pandas as pd

from task3.config import CURRENCY_TO_USD
from task3.features.parsers import parse_remote, parse_size, parse_skills, seniority


def preprocess_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the jobs DataFrame in-place and return it."""
    jobs = jobs.copy()

    # ── Salary normalisation ───────────────────────────────────────────────────
    jobs["salary_currency"] = jobs["salary_currency"].str.strip().str.upper()
    jobs["salary_min"] = pd.to_numeric(jobs["salary_min"], errors="coerce").fillna(0.0)
    jobs["salary_max"] = pd.to_numeric(jobs["salary_max"], errors="coerce").fillna(0.0)

    cur_map = jobs["salary_currency"].map(CURRENCY_TO_USD).fillna(1.0)
    jobs["salary_mid_usd"] = ((jobs["salary_min"] + jobs["salary_max"]) / 2) * cur_map
    jobs["salary_max_usd"] = jobs["salary_max"] * cur_map

    # ── Categorical → numeric helpers ──────────────────────────────────────────
    jobs["remote_num"] = jobs["remote_allowed"].apply(parse_remote)
    jobs["company_size_num"] = jobs["company_size"].apply(parse_size)
    jobs["job_seniority"] = jobs["job_title"].apply(seniority)

    # ── Skills as sets ─────────────────────────────────────────────────────────
    jobs["req_skills_set"] = jobs["required_skills"].apply(parse_skills)

    return jobs

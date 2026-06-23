import os
import numpy as np

SEED = 42
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
                 "..", "Datasets", "task 3 - datasets")
)

JOBS_PATH = os.path.join(_DATA_DIR, "jobs.csv")
CANDIDATES_PATH = os.path.join(_DATA_DIR, "candidates.csv")
TRAIN_PATH = os.path.join(_DATA_DIR, "applications_train.csv")
TEST_PATH = os.path.join(_DATA_DIR, "applications_test.csv")
SUBMISSION_PATH = "task3_submission.csv"

# ── Currency → USD rates ──────────────────────────────────────────────────────
CURRENCY_TO_USD = {"USD": 1.0, "EUR": 1.08, "IRR": 0.000024}

# ── Feature columns used by the model ─────────────────────────────────────────
FEATURE_COLS = [
    "skill_overlap",
    "skill_coverage",
    "num_matching_skills",
    "req_skills_count",
    "exp_gap_min",
    "exp_in_range",
    "salary_ratio",
    "salary_affordable",
    "seniority_match",
    "seniority_diff",
    "edu_num",
    "eng_num",
    "cert_count",
    "prev_companies_count",
    "willing_to_relocate_bin",
    "profile_completeness",
    "location_compat",
    "remote_num",
    "company_size_num",
    "job_seniority",
    "cand_seniority",
    "years_experience",
    "expected_salary",
    "days_since_posted",
    "app_month",
    "industry_enc",
]

# ── Subset columns for merge ───────────────────────────────────────────────────
JOB_COLS = [
    "job_id",
    "req_skills_set",
    "min_years_experience",
    "max_years_experience",
    "salary_mid_usd",
    "salary_max_usd",
    "remote_num",
    "company_size_num",
    "job_seniority",
    "job_posted_date",
    "industry",
    "job_location",
]

CAND_COLS = [
    "candidate_id",
    "years_experience",
    "edu_num",
    "eng_num",
    "cert_count",
    "prev_companies_count",
    "willing_to_relocate_bin",
    "expected_salary",
    "profile_completeness",
    "cand_skills_set",
    "cand_seniority",
    "candidate_location",
]

# ── Cross-validation ──────────────────────────────────────────────────────────
N_SPLITS = 5

# ── Model hyper-parameters ────────────────────────────────────────────────────
MODEL_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=SEED,
)

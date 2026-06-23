import pandas as pd

from task3.features.parsers import parse_edu, parse_eng, parse_skills, seniority


def preprocess_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the candidates DataFrame in-place and return it."""
    candidates = candidates.copy()

    candidates["edu_num"] = candidates["education_level"].apply(parse_edu)
    candidates["eng_num"] = candidates["english_proficiency"].apply(parse_eng)

    candidates["years_experience"] = pd.to_numeric(
        candidates["years_experience"], errors="coerce"
    )
    candidates["years_experience"] = candidates["years_experience"].fillna(
        candidates["years_experience"].median()
    )

    candidates["expected_salary"] = pd.to_numeric(
        candidates["expected_salary"], errors="coerce"
    ).fillna(0.0)

    candidates["profile_completeness"] = pd.to_numeric(
        candidates["profile_completeness"], errors="coerce"
    ).fillna(50.0)

    candidates["cand_skills_set"] = candidates["skills"].apply(parse_skills)

    candidates["cert_count"] = candidates["certifications"].apply(
        lambda x: len(str(x).split("|")) if pd.notna(x) and str(x).strip() else 0
    )
    candidates["prev_companies_count"] = candidates["previous_companies"].apply(
        lambda x: len(str(x).split("|")) if pd.notna(x) and str(x).strip() else 0
    )
    candidates["willing_to_relocate_bin"] = candidates["willing_to_relocate"].apply(
        lambda x: 1 if str(x).strip().lower() in ("yes", "true", "1") else 0
    )
    candidates["cand_seniority"] = candidates["current_title"].apply(seniority)

    return candidates

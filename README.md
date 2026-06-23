# Task 3 — Learning-to-Rank for Job-Candidate Matching

A machine learning pipeline that predicts relevance scores between job postings and candidate profiles, ranking candidates for each job opening.

## Project Structure

```
task3/
├── __main__.py            # Entry point (python -m task3)
├── config.py              # Paths, feature list, model hyperparameters
├── pipeline.py            # End-to-end orchestration
├── data/
│   └── loader.py          # CSV data loading
├── preprocessing/
│   ├── jobs.py            # Salary normalization, categorical encoding, skill parsing
│   └── candidates.py      # Education, experience, skill preprocessing
├── features/
│   ├── parsers.py         # Utility parsers (skills, seniority, education, etc.)
│   ├── interactions.py    # Skill overlap, experience gap, salary/compatibility features
│   └── encoding.py        # Out-of-fold target encoding for industry
├── models/
│   └── ranker.py          # GradientBoostingRegressor with GroupKFold CV
├── evaluation/
│   └── metrics.py         # NDCG@10 and MAP@5 evaluation metrics
└── datasets/              # Raw input data (CSV files)
    ├── jobs.csv
    ├── candidates.csv
    ├── applications_train.csv
    └── applications_test.csv
```

## Datasets

| File | Description |
|---|---|
| `jobs.csv` | Job postings with title, required skills, salary, location, remote policy, company size |
| `candidates.csv` | Candidate profiles with skills, experience, education, English proficiency, salary expectations |
| `applications_train.csv` | Training set — candidate–job pairs with relevance labels |
| `applications_test.csv` | Test set — candidate–job pairs to score |

## Features

The model uses 25 engineered features:

- **Skill-based**: skill overlap (Jaccard), skill coverage, number of matching skills
- **Experience**: experience gap, whether candidate is in the required range
- **Salary**: salary ratio (expected / mid), affordability flag
- **Seniority**: match flag and difference between candidate and job seniority
- **Profile**: education level, English proficiency, certification count, previous companies
- **Location**: compatibility heuristic (remote, relocation, city match)
- **Temporal**: days since job posted, application month
- **Encoded**: industry (out-of-fold target encoding)

## Model

- **Algorithm**: GradientBoostingRegressor (scikit-learn)
- **Validation**: GroupKFold cross-validation (5 folds, grouped by job_id)
- **Evaluation Metrics**: NDCG@10, MAP@5
- **Hyperparameters**: 200 estimators, learning rate 0.05, max depth 5

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
```

## Usage

```bash
# From the project root directory (one level above task3/)
python -m task3
```

The pipeline will:

1. Load the four datasets from the configured path
2. Preprocess jobs and candidates (clean, encode, parse skills)
3. Build interaction features between candidates and jobs
4. Apply target encoding for the industry column
5. Run 5-fold GroupKFold cross-validation and print NDCG@10 / MAP@5
6. Train a final model on the full training set
7. Predict scores on the test set and save `task3_submission.csv`

## Output

The pipeline produces `task3_submission.csv` with columns `application_id` and `score`, sorted by application ID.

## Student IDs

- 810103482
- 810103601
- 810103434

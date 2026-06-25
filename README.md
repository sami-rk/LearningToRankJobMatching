# Task 3 — Learning-to-Rank for Job-Candidate Matching

A machine learning pipeline that predicts relevance scores between job postings and candidate profiles, ranking candidates for each job opening.

## Project Structure

```
LearningToRankJobMatching/
├── __init__.py            # Package marker
├── __main__.py            # Entry-point for python -m
├── config.py              # Paths, feature list, model hyperparameters
├── pipeline.py            # End-to-end orchestration (entry point)
├── data/
│   ├── __init__.py
│   └── loader.py          # CSV data loading
├── preprocessing/
│   ├── __init__.py
│   ├── jobs.py            # Salary normalization, categorical encoding, skill parsing
│   └── candidates.py      # Education, experience, skill preprocessing
├── features/
│   ├── __init__.py
│   ├── parsers.py         # Utility parsers (skills, seniority, education, etc.)
│   ├── interactions.py    # Skill overlap, experience gap, salary/compatibility features
│   └── encoding.py        # Out-of-fold target encoding for industry
├── models/
│   ├── __init__.py
│   └── ranker.py          # Multi-model ranker (GradientBoosting, XGBoost, LightGBM, CatBoost)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py         # NDCG@10 and MAP@5 evaluation metrics
│   └── visuals.py         # Comparison plots (bar, box, radar)
├── output/                # Generated plots
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

The model uses 26 engineered features:

- **Skill-based**: skill overlap (Jaccard), skill coverage, number of matching skills
- **Experience**: experience gap, whether candidate is in the required range
- **Salary**: salary ratio (expected / mid), affordability flag
- **Seniority**: match flag and difference between candidate and job seniority
- **Profile**: education level, English proficiency, certification count, previous companies
- **Location**: compatibility heuristic (remote, relocation, city match)
- **Temporal**: days since job posted, application month
- **Encoded**: industry (out-of-fold target encoding)

## Models Compared

The pipeline compares four learning-to-rank approaches:

| Model | Approach | Objective | Description |
|---|---|---|---|
| **GradientBoosting** | Pointwise | Regression | Baseline — predicts relevance scores directly |
| **XGBoost** | Pairwise | `rank:ndcg` | Optimizes pairwise ranking with NDCG objective |
| **LightGBM** | Listwise | `lambdarank` | LambdaRank — fast, histogram-based listwise ranking |
| **CatBoost** | Listwise | `YetiRank` | Ordered boosting with YetiRank objective |

### Validation Strategy

- **Method**: GroupKFold cross-validation (5 folds, grouped by `job_id`)
- **Metrics**: NDCG@10, MAP@5
- **Best model**: Selected by highest mean NDCG@10 across folds, used for final submission

## Visualizations

The pipeline generates three comparison plots in the `output/` directory:

| Plot | Description |
|---|---|
| `metric_comparison.png` | Grouped bar chart with NDCG@10 and MAP@5 per model (with error bars) |
| `fold_distribution.png` | Box plots showing per-fold metric variance for each model |
| `radar_chart.png` | Spider/radar chart for multi-metric comparison across models |

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
matplotlib>=3.7
seaborn>=0.12
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# From the parent directory (one level above LearningToRankJobMatching/)
python -m LearningToRankJobMatching.pipeline
```

The pipeline will:

1. Load the four datasets from the configured path
2. Preprocess jobs and candidates (clean, encode, parse skills)
3. Build interaction features between candidates and jobs
4. Apply target encoding for the industry column
5. Run 5-fold GroupKFold cross-validation for all four models
6. Print NDCG@10 and MAP@5 results for each model
7. Generate comparison plots in `output/`
8. Train the best model on the full training set
9. Predict scores on the test set and save `task3_submission.csv`

## Results

| Dataset | Shape |
|---|---|
| X_train | 118,772 × 26 |
| X_test | 52,700 × 26 |

| Model | NDCG@10 | MAP@5 |
|---|---|---|
| GradientBoosting | 0.8463 | 0.6589 |
| XGBoost | — | — |
| LightGBM | — | — |
| CatBoost | — | — |

> Results for XGBoost, LightGBM, and CatBoost will be populated after the first pipeline run.

## Output

The pipeline produces:
- `task3_submission.csv` — columns `application_id` and `score`, sorted by application ID
- `output/metric_comparison.png` — bar chart of model metrics
- `output/fold_distribution.png` — box plots of fold-wise metric distribution
- `output/radar_chart.png` — radar chart of multi-metric comparison

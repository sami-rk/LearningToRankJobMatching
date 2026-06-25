import pandas as pd

from LearningToRankJobMatching.config import JOBS_PATH, CANDIDATES_PATH, TRAIN_PATH, TEST_PATH


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and return (jobs, candidates, train, test)."""
    jobs = pd.read_csv(JOBS_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Loaded  jobs={jobs.shape}  candidates={candidates.shape}  "
          f"train={train.shape}  test={test.shape}")
    return jobs, candidates, train, test

"""Entry-point: ``python -m LearningToRankJobMatching``."""

import warnings

warnings.filterwarnings("ignore")

from LearningToRankJobMatching.pipeline import run

if __name__ == "__main__":
    run()

"""Entry-point: ``python -m LearningToRankJobMatching``."""

import argparse
import warnings

warnings.filterwarnings("ignore")

from LearningToRankJobMatching.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Learning-to-Rank pipeline")
    parser.add_argument(
        "--retrain", action="store_true", default=False,
        help="Force retrain all models (ignore cached results)",
    )
    args = parser.parse_args()
    run(force_retrain=args.retrain)


if __name__ == "__main__":
    main()

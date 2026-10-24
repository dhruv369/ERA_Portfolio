"""Run the complete Session 2 multilingual BPE pipeline."""

import argparse
from pathlib import Path

import clean_articles
import evaluate_bpe
import fetch_articles
import train_bpe


DATA_PATH = Path(__file__).resolve().parent / "data" / "articles.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Wikipedia articles, train a shared BPE tokenizer, and evaluate it."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download the latest Wikipedia articles before running the pipeline.",
    )
    args = parser.parse_args()

    if args.refresh or not DATA_PATH.exists():
        print("Step 1/4: Downloading Wikipedia articles")
        fetch_articles.main()
    else:
        print(f"Step 1/4: Reusing downloaded articles from {DATA_PATH}")

    print("\nStep 2/4: Cleaning article text")
    clean_articles.main()

    print("\nStep 3/4: Training the shared BPE tokenizer")
    train_bpe.main()

    print("\nStep 4/4: Evaluating tokenization")
    evaluate_bpe.main()


if __name__ == "__main__":
    main()

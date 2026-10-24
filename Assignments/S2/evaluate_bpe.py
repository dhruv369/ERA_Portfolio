"""Evaluate the shared BPE tokenizer on each cleaned language article."""

import argparse
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


DATA_PATH = Path(__file__).resolve().parent / "data" / "cleaned_articles.json"
TOKENIZER_DIR = Path(__file__).resolve().parent
EXPECTED_LANGUAGES = {"en", "hi", "te", "ta"}
EXPECTED_VOCAB_SIZE = 10_000


def load_articles() -> list[dict[str, str]]:
    with DATA_PATH.open(encoding="utf-8") as source_file:
        data: Any = json.load(source_file)

    if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
        raise ValueError(f"Unexpected article data format in {DATA_PATH}")

    articles: list[dict[str, str]] = []
    found_languages: set[str] = set()
    for article in data["articles"]:
        if not isinstance(article, dict):
            raise ValueError(f"Unexpected article entry in {DATA_PATH}")

        language = article.get("language")
        language_name = article.get("language_name")
        text = article.get("text")
        if not all(isinstance(value, str) and value for value in (language, language_name, text)):
            raise ValueError(f"Article is missing language, name, or text in {DATA_PATH}")
        if language in found_languages:
            raise ValueError(f"Duplicate article for language {language}")

        found_languages.add(language)
        articles.append(
            {"language": language, "language_name": language_name, "text": text}
        )

    if found_languages != EXPECTED_LANGUAGES:
        missing = ", ".join(sorted(EXPECTED_LANGUAGES - found_languages))
        unexpected = ", ".join(sorted(found_languages - EXPECTED_LANGUAGES))
        details = []
        if missing:
            details.append(f"missing: {missing}")
        if unexpected:
            details.append(f"unexpected: {unexpected}")
        raise ValueError("Expected one article per language (" + "; ".join(details) + ")")

    return articles


def evaluate(experiment: str) -> None:
    tokenizer_path = TOKENIZER_DIR / f"tokenizer_{experiment}.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Expected a {EXPECTED_VOCAB_SIZE}-entry tokenizer, found {vocab_size}"
        )

    results: list[dict[str, Any]] = []
    for article in load_articles():
        words = article["text"].split()
        if not words:
            raise ValueError(f"Article has no words: {article['language']}")

        unique_words = set(words)
        token_count = sum(len(tokenizer.encode(word).ids) for word in unique_words)
        unique_word_count = len(unique_words)
        ratio = token_count / unique_word_count
        results.append(
            {
                "language": article["language"],
                "language_name": article["language_name"],
                "token_count": token_count,
                "unique_word_count": unique_word_count,
                "ratio": ratio,
            }
        )

    lowest = min(results, key=lambda result: result["ratio"])
    highest = max(results, key=lambda result: result["ratio"])
    ratio_gap = highest["ratio"] - lowest["ratio"]
    if ratio_gap == 0:
        raise ValueError("The score is undefined because all language ratios are equal")
    score = 1000 / ratio_gap

    print(f"Experiment: {experiment}")
    print(f"Tokenizer vocabulary: {vocab_size:,} entries")
    print(
        "Each distinct whitespace-separated form is encoded once; "
        "case and punctuation are kept."
    )
    print()
    print(f"{'Language':<10} {'Unique-word tokens':>19} {'Unique words':>14} {'X_i':>10}")
    for result in results:
        print(
            f"{result['language_name']:<10} "
            f"{result['token_count']:>10,} "
            f"{result['unique_word_count']:>14,} "
            f"{result['ratio']:>10.4f}"
        )
    print()
    print(
        f"Ratio gap: {highest['ratio']:.4f} - {lowest['ratio']:.4f} "
        f"= {ratio_gap:.4f}"
    )
    print(f"Score: 1000 / {ratio_gap:.4f} = {score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an experiment tokenizer.")
    parser.add_argument("--experiment", choices=("A", "B", "C"), default="A")
    args = parser.parse_args()
    evaluate(args.experiment)


if __name__ == "__main__":
    main()

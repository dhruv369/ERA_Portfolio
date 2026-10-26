"""Compare reserved Indic-script subword slots against Experiment D."""

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

import evaluate_bpe
import train_bpe


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "reserved_vocab_results.json"
RESERVATION_SIZES = (250, 500, 1_000)
INDIC_LANGUAGES = ("hi", "te", "ta")
SCRIPT_NAMES = {
    "hi": "DEVANAGARI",
    "te": "TELUGU",
    "ta": "TAMIL",
}
TRAINING_WEIGHTS = {"en": 1, "hi": 2, "te": 4, "ta": 6}
MAX_SUBWORD_UNITS = 5


def indic_subword_frequencies(text: str, script_name: str) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for word in text.split():
        script_runs: list[str] = []
        current_run = ""
        for character in word:
            name = unicodedata.name(character, "")
            is_script_character = script_name in name
            is_mark = unicodedata.category(character).startswith("M")
            if is_script_character or (is_mark and current_run):
                current_run += character
            else:
                if current_run:
                    script_runs.append(current_run)
                    current_run = ""
        if current_run:
            script_runs.append(current_run)

        for run in script_runs:
            units: list[str] = []
            for character in run:
                if unicodedata.category(character).startswith("M") and units:
                    units[-1] += character
                else:
                    units.append(character)

            for start in range(len(units)):
                max_end = min(len(units), start + MAX_SUBWORD_UNITS)
                for end in range(start + 2, max_end + 1):
                    if start == 0 and end == len(units):
                        continue
                    candidate = "".join(units[start:end])
                    letter_count = sum(
                        unicodedata.category(character).startswith("L")
                        for character in candidate
                    )
                    if letter_count >= 2:
                        frequencies[candidate] += 1
    return frequencies


def build_candidate_groups(
    texts_by_language: dict[str, str],
) -> dict[str, list[str]]:
    candidates_by_language: dict[str, list[str]] = {}
    for language in INDIC_LANGUAGES:
        frequencies = indic_subword_frequencies(
            texts_by_language[language], SCRIPT_NAMES[language]
        )
        candidates_by_language[language] = [
            candidate
            for candidate, _ in sorted(
                frequencies.items(),
                key=lambda item: (-item[1], len(item[0]), item[0]),
            )
        ]
    return candidates_by_language


def evaluate_tokenizer(tokenizer: Tokenizer, label: str) -> dict[str, Any]:
    results, ratio_gap, score = evaluate_bpe.measure(tokenizer)
    return {
        "label": label,
        "vocabulary_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "ratio_gap": ratio_gap,
        "score": score,
        "languages": results,
    }


def load_revision_ids() -> dict[str, int]:
    with train_bpe.DATA_PATH.open(encoding="utf-8") as source_file:
        data: Any = json.load(source_file)

    if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
        raise ValueError(f"Unexpected article data format in {train_bpe.DATA_PATH}")

    revision_ids = {
        article.get("language"): article.get("revision_id")
        for article in data["articles"]
        if isinstance(article, dict)
    }
    if set(revision_ids) != train_bpe.LANGUAGES or not all(
        isinstance(revision_id, int) for revision_id in revision_ids.values()
    ):
        raise ValueError(
            f"Expected revision IDs for {', '.join(sorted(train_bpe.LANGUAGES))}"
        )
    return revision_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Indic-script reserved vocabulary with a fixed 10,000-token budget."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=RESERVATION_SIZES,
        help="Numbers of vocabulary entries reserved for Indic subwords.",
    )
    args = parser.parse_args()
    if any(size <= 0 or size >= train_bpe.VOCAB_SIZE for size in args.sizes):
        parser.error(
            f"each reservation size must be between 1 and "
            f"{train_bpe.VOCAB_SIZE - 1}"
        )
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("reservation sizes must be unique")

    articles = evaluate_bpe.load_articles()
    revision_ids = load_revision_ids()
    texts_by_language = {article["language"]: article["text"] for article in articles}
    training_texts = [
        texts_by_language[language]
        for language, weight in TRAINING_WEIGHTS.items()
        for _ in range(weight)
    ]
    candidates_by_language = build_candidate_groups(texts_by_language)

    baseline_tokenizer = Tokenizer.from_file(str(HERE / "tokenizer_D.json"))
    baseline = evaluate_tokenizer(baseline_tokenizer, "D baseline")
    experiments = [baseline]
    print(f"D baseline: score={baseline['score']:.2f}")

    for reserved_size in args.sizes:
        tokenizer, reserved_counts = (
            train_bpe.train_tokenizer_with_reserved_indic_tokens(
                training_texts,
                candidates_by_language,
                reserved_size,
            )
        )
        label = f"E-{reserved_size}"
        tokenizer_path = HERE / f"tokenizer_{label}.json"
        tokenizer.save(str(tokenizer_path))

        result = evaluate_tokenizer(tokenizer, label)
        result["reserved_token_count"] = reserved_size
        result["reserved_tokens_by_language"] = reserved_counts
        result["tokenizer_file"] = tokenizer_path.name
        experiments.append(result)
        print(
            f"{label}: score={result['score']:.2f}, "
            f"ratio gap={result['ratio_gap']:.4f}, "
            f"reserved={reserved_counts}"
        )

    output = {
        "training_weights": "1:2:4:6 (English:Hindi:Telugu:Tamil)",
        "vocabulary_size": train_bpe.VOCAB_SIZE,
        "score_formula": "1000 / (max(X_i) - min(X_i))",
        "reservation_method": (
            "Reserve equally across Hindi, Telugu, and Tamil by adding frequent "
            "Indic-script substrings as normal, non-special tokenizer entries; "
            "train the remaining BPE vocabulary on weighted article text."
        ),
        "candidate_method": (
            "Count NFC article substrings of 2-5 script units that occur inside "
            "a word; retain candidates with at least two Unicode letters."
        ),
        "revision_ids": {
            language: revision_ids[language] for language in train_bpe.LANGUAGE_ORDER
        },
        "experiments": experiments,
    }
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved experiment results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

"""Train one shared byte-level BPE tokenizer on the four cleaned articles."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

import evaluate_bpe


DATA_PATH = Path(__file__).resolve().parent / "data" / "cleaned_articles.json"
OUTPUT_DIR = Path(__file__).resolve().parent
LANGUAGES = {"en", "hi", "te", "ta"}
LANGUAGE_ORDER = ("en", "hi", "te", "ta")
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]
RANDOM_SEED = 42
D_WEIGHT_CANDIDATES = (
    (1, 1, 1, 1),
    (1, 1, 1, 2),
    (1, 1, 2, 2),
    (1, 2, 2, 2),
    (1, 2, 3, 4),
    (1, 2, 3, 6),
    (1, 2, 4, 6),
)


def load_training_texts(experiment: str) -> list[str]:
    with DATA_PATH.open(encoding="utf-8") as source_file:
        data: Any = json.load(source_file)

    if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
        raise ValueError(f"Unexpected article data format in {DATA_PATH}")

    articles = data["articles"]
    texts_by_language: dict[str, str] = {}
    found_languages: set[str] = set()
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError(f"Unexpected article entry in {DATA_PATH}")

        language = article.get("language")
        text = article.get("text")
        if not isinstance(language, str) or not isinstance(text, str) or not text:
            raise ValueError(f"Article is missing language or text in {DATA_PATH}")
        if language in found_languages:
            raise ValueError(f"Duplicate article for language {language}")

        found_languages.add(language)
        texts_by_language[language] = text

    if found_languages != LANGUAGES:
        missing = ", ".join(sorted(LANGUAGES - found_languages))
        unexpected = ", ".join(sorted(found_languages - LANGUAGES))
        details = []
        if missing:
            details.append(f"missing: {missing}")
        if unexpected:
            details.append(f"unexpected: {unexpected}")
        raise ValueError("Expected one article per language (" + "; ".join(details) + ")")

    if experiment in {"A", "D"}:
        return [texts_by_language[language] for language in LANGUAGE_ORDER]
    if experiment not in {"B", "C"}:
        raise ValueError(f"Unknown experiment: {experiment}")

    word_lists = {
        language: texts_by_language[language].split()
        for language in LANGUAGE_ORDER
    }
    if experiment == "B":
        target_word_count = max(len(words) for words in word_lists.values())
    else:
        target_character_count = max(
            len(texts_by_language[language]) for language in LANGUAGE_ORDER
        )

    balanced_texts = []
    for index, language in enumerate(LANGUAGE_ORDER):
        words = word_lists[language]
        rng = random.Random(RANDOM_SEED + index)
        if experiment == "B":
            extra_words = rng.choices(words, k=target_word_count - len(words))
            training_text = " ".join(words + extra_words)
            print(
                f"{language}: training with {target_word_count:,} words "
                f"({len(words):,} original, {len(extra_words):,} sampled)"
            )
        else:
            extra_words = []
            character_count = len(texts_by_language[language])
            while character_count < target_character_count:
                sampled_word = rng.choice(words)
                extra_words.append(sampled_word)
                character_count += len(sampled_word) + 1
            training_text = " ".join(words + extra_words)
            print(
                f"{language}: training with {len(training_text):,} characters "
                f"({len(texts_by_language[language]):,} original, "
                f"{len(extra_words):,} sampled words)"
            )
        balanced_texts.append(training_text)

    return balanced_texts


def train(experiment: str) -> Path:
    tokenizer = train_tokenizer(load_training_texts(experiment))
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    tokenizer_path = OUTPUT_DIR / f"tokenizer_{experiment}.json"
    tokenizer.save(str(tokenizer_path))
    print(
        f"Saved Experiment {experiment} tokenizer with "
        f"{actual_vocab_size} entries to {tokenizer_path}"
    )
    print(f"Special tokens ({len(SPECIAL_TOKENS)}): {', '.join(SPECIAL_TOKENS)}")
    return tokenizer_path


def train_tokenizer(training_texts: list[str]) -> Tokenizer:
    return _train_tokenizer(training_texts, VOCAB_SIZE)


def train_tokenizer_with_reserved_indic_tokens(
    training_texts: list[str],
    candidates_by_language: dict[str, list[str]],
    reserved_token_count: int,
) -> tuple[Tokenizer, dict[str, int]]:
    if reserved_token_count <= 0 or reserved_token_count >= VOCAB_SIZE:
        raise ValueError(
            f"Reserved token count must be between 1 and {VOCAB_SIZE - 1}"
        )
    if not candidates_by_language:
        raise ValueError("At least one Indic language candidate list is required")

    languages = tuple(candidates_by_language)
    base_quota, remainder = divmod(reserved_token_count, len(languages))
    quotas = {
        language: base_quota + int(index < remainder)
        for index, language in enumerate(languages)
    }
    tokenizer = _train_tokenizer(training_texts, VOCAB_SIZE - reserved_token_count)
    existing_tokens = tokenizer.get_vocab()
    reserved_tokens: list[AddedToken] = []
    reserved_counts: dict[str, int] = {}
    reserved_values: set[str] = set()

    for language, candidates in candidates_by_language.items():
        selected: list[str] = []
        if quotas[language] == 0:
            reserved_counts[language] = 0
            continue
        for candidate in candidates:
            if candidate not in existing_tokens and candidate not in reserved_values:
                selected.append(candidate)
                reserved_values.add(candidate)
                if len(selected) == quotas[language]:
                    break
        if len(selected) != quotas[language]:
            raise RuntimeError(
                f"Not enough unused Indic candidates for {language}: "
                f"needed {quotas[language]}, found {len(selected)}"
            )
        reserved_tokens.extend(
            AddedToken(token, normalized=True, special=False) for token in selected
        )
        reserved_counts[language] = len(selected)

    added_count = tokenizer.add_tokens(reserved_tokens)
    if added_count != reserved_token_count:
        raise RuntimeError(
            f"Expected to add {reserved_token_count} reserved Indic tokens, "
            f"but added {added_count}"
        )

    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected exactly {VOCAB_SIZE} vocabulary entries, "
            f"but training produced {actual_vocab_size}"
        )
    return tokenizer, reserved_counts


def _train_tokenizer(training_texts: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(training_texts, trainer=trainer)
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != vocab_size:
        raise RuntimeError(
            f"Expected exactly {vocab_size} vocabulary entries, "
            f"but training produced {actual_vocab_size}"
        )
    return tokenizer


def run_weight_search() -> Path:
    articles = load_training_texts("D")
    candidates: list[tuple[tuple[int, int, int, int], Tokenizer, float]] = []

    print("Experiment D: deterministic integer sampling-weight search")
    for weights in D_WEIGHT_CANDIDATES:
        training_texts = [
            text
            for index, text in enumerate(articles)
            for _ in range(weights[index])
        ]
        tokenizer = train_tokenizer(training_texts)
        _, ratio_gap, score = evaluate_bpe.measure(tokenizer)
        candidates.append((weights, tokenizer, score))
        print(
            f"Weights en:hi:te:ta={':'.join(map(str, weights))} "
            f"| D={ratio_gap:.4f} | score={score:.2f}"
        )

    best_weights, best_tokenizer, best_score = max(candidates, key=lambda item: item[2])
    tokenizer_path = OUTPUT_DIR / "tokenizer_D.json"
    best_tokenizer.save(str(tokenizer_path))
    print(
        f"Best weights en:hi:te:ta={':'.join(map(str, best_weights))}; "
        f"score={best_score:.2f}"
    )
    print(f"Saved best Experiment D tokenizer to {tokenizer_path}")
    return tokenizer_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shared BPE tokenizer.")
    parser.add_argument("--experiment", choices=("A", "B", "C", "D"), default="A")
    args = parser.parse_args()
    if args.experiment == "D":
        run_weight_search()
    else:
        train(args.experiment)


if __name__ == "__main__":
    main()

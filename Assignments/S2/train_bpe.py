"""Train one shared byte-level BPE tokenizer on the four cleaned articles."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


DATA_PATH = Path(__file__).resolve().parent / "data" / "cleaned_articles.json"
OUTPUT_DIR = Path(__file__).resolve().parent
LANGUAGES = {"en", "hi", "te", "ta"}
LANGUAGE_ORDER = ("en", "hi", "te", "ta")
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]
RANDOM_SEED = 42


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

    if experiment == "A":
        return [texts_by_language[language] for language in LANGUAGE_ORDER]
    if experiment != "B":
        raise ValueError(f"Unknown experiment: {experiment}")

    word_lists = {
        language: texts_by_language[language].split()
        for language in LANGUAGE_ORDER
    }
    target_word_count = max(len(words) for words in word_lists.values())
    balanced_texts = []
    for index, language in enumerate(LANGUAGE_ORDER):
        words = word_lists[language]
        rng = random.Random(RANDOM_SEED + index)
        extra_words = rng.choices(words, k=target_word_count - len(words))
        balanced_texts.append(" ".join(words + extra_words))
        print(
            f"{language}: training with {target_word_count:,} words "
            f"({len(words):,} original, {len(extra_words):,} sampled)"
        )

    return balanced_texts


def train(experiment: str) -> Path:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(load_training_texts(experiment), trainer=trainer)

    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected exactly {VOCAB_SIZE} vocabulary entries, "
            f"but training produced {actual_vocab_size}"
        )

    tokenizer_path = OUTPUT_DIR / f"tokenizer_{experiment}.json"
    tokenizer.save(str(tokenizer_path))
    print(
        f"Saved Experiment {experiment} tokenizer with "
        f"{actual_vocab_size} entries to {tokenizer_path}"
    )
    print(f"Special tokens ({len(SPECIAL_TOKENS)}): {', '.join(SPECIAL_TOKENS)}")
    return tokenizer_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shared BPE tokenizer.")
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    args = parser.parse_args()
    train(args.experiment)


if __name__ == "__main__":
    main()

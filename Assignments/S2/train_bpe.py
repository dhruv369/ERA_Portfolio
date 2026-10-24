"""Train one shared byte-level BPE tokenizer on the four cleaned articles."""

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


DATA_PATH = Path(__file__).resolve().parent / "data" / "cleaned_articles.json"
TOKENIZER_PATH = Path(__file__).resolve().parent / "tokenizer.json"
LANGUAGES = {"en", "hi", "te", "ta"}
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]


def load_training_texts() -> list[str]:
    with DATA_PATH.open(encoding="utf-8") as source_file:
        data: Any = json.load(source_file)

    if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
        raise ValueError(f"Unexpected article data format in {DATA_PATH}")

    articles = data["articles"]
    texts: list[str] = []
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
        texts.append(text)

    if found_languages != LANGUAGES:
        missing = ", ".join(sorted(LANGUAGES - found_languages))
        unexpected = ", ".join(sorted(found_languages - LANGUAGES))
        details = []
        if missing:
            details.append(f"missing: {missing}")
        if unexpected:
            details.append(f"unexpected: {unexpected}")
        raise ValueError("Expected one article per language (" + "; ".join(details) + ")")

    return texts


def main() -> None:
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
    tokenizer.train_from_iterator(load_training_texts(), trainer=trainer)

    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected exactly {VOCAB_SIZE} vocabulary entries, "
            f"but training produced {actual_vocab_size}"
        )

    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Saved shared BPE tokenizer with {actual_vocab_size} entries to {TOKENIZER_PATH}")
    print(f"Special tokens ({len(SPECIAL_TOKENS)}): {', '.join(SPECIAL_TOKENS)}")


if __name__ == "__main__":
    main()

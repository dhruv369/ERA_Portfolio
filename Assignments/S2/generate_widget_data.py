"""Export verified Experiment D metrics and vocabulary for the static widget."""

import json
import shutil
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

import evaluate_bpe


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "cleaned_articles.json"
TOKENIZER_PATH = HERE / "tokenizer_D.json"
WIDGET_DIR = HERE / "widget"
RESULTS_PATH = WIDGET_DIR / "results.json"
VOCAB_PATH = WIDGET_DIR / "tokenizer-vocab.tsv"
TOKENIZER_DOWNLOAD_PATH = WIDGET_DIR / "tokenizer_D.json"
EXPECTED_VOCAB_SIZE = 10_000
WEIGHTS = "1:2:4:6 (English:Hindi:Telugu:Tamil)"


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as source_file:
        data: Any = json.load(source_file)

    if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
        raise ValueError(f"Unexpected article data format in {DATA_PATH}")

    articles_by_language = {
        article.get("language"): article
        for article in data["articles"]
        if isinstance(article, dict)
    }
    if set(articles_by_language) != evaluate_bpe.EXPECTED_LANGUAGES:
        raise ValueError("Widget export requires exactly the four evaluated articles")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    if len(vocab) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Expected {EXPECTED_VOCAB_SIZE} vocabulary entries, found {len(vocab)}")

    results, ratio_gap, score = evaluate_bpe.measure(tokenizer)
    languages = []
    for result in results:
        source_article = articles_by_language[result["language"]]
        languages.append(
            {
                **result,
                "page_url": source_article["page_url"],
                "revision_id": source_article["revision_id"],
            }
        )

    output = {
        "experiment": "D",
        "training_weights": WEIGHTS,
        "vocabulary_size": len(vocab),
        "special_tokens": ["<|pad|>", "<|bos|>", "<|eos|>"],
        "word_counting_rule": (
            "Split cleaned article text on whitespace; retain case and punctuation; "
            "encode each distinct form once."
        ),
        "score_formula": "1000 / (max(X_i) - min(X_i))",
        "ratio_gap": ratio_gap,
        "score": score,
        "languages": languages,
    }

    WIDGET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TOKENIZER_PATH, TOKENIZER_DOWNLOAD_PATH)
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    vocab_lines = [
        f"{token_id}\t{json.dumps(token, ensure_ascii=False)}"
        for token, token_id in sorted(vocab.items(), key=lambda item: item[1])
    ]
    VOCAB_PATH.write_text("\n".join(vocab_lines) + "\n", encoding="utf-8")

    print(f"Verified {len(vocab):,} vocabulary entries")
    print(f"Saved widget metrics to {RESULTS_PATH}")
    print(f"Saved token ID list to {VOCAB_PATH}")
    print(f"Copied complete tokenizer to {TOKENIZER_DOWNLOAD_PATH}")
    print(f"Verified score: D={ratio_gap:.8f}, score={score:.8f}")


if __name__ == "__main__":
    main()

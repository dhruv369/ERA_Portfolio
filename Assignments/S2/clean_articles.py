"""Normalize the downloaded Wikipedia article text for tokenizer experiments."""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_PATH = DATA_DIR / "articles.json"
OUTPUT_PATH = DATA_DIR / "cleaned_articles.json"
HEADING_PATTERN = re.compile(r"(?m)^={2,}\s*(.*?)\s*={2,}\s*$")


def normalize_text(text: str) -> str:
    """Normalize Unicode, remove plaintext heading markers, and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = HEADING_PATTERN.sub(r"\1", text)
    return " ".join(text.split())


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8") as source_file:
        data: Any = json.load(source_file)

    if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
        raise ValueError(f"Unexpected article data format in {INPUT_PATH}")

    for article in data["articles"]:
        if not isinstance(article, dict) or not isinstance(article.get("text"), str):
            raise ValueError(f"Unexpected article entry in {INPUT_PATH}")
        article["text"] = normalize_text(article["text"])

    data["text_cleaning"] = [
        "Unicode NFC normalization",
        "Wikipedia plaintext heading marker removal",
        "Whitespace collapse",
    ]
    OUTPUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved normalized articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

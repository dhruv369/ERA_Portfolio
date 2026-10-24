# Session 2: A multilingual BPE tokenizer

## Experiment

Build a Byte Pair Encoding (BPE) tokenizer with one shared vocabulary of 10,000 tokens for the English, Hindi, Telugu, and Tamil versions of Wikipedia's India article.

## Prepare the articles

Run `python Assignments/S2/fetch_articles.py` to collect the corresponding India articles from Wikidata and Wikipedia. The script saves their plain text, page URLs, and revision IDs to `Assignments/S2/data/articles.json`. This downloaded corpus is local and ignored by Git.

Run `python Assignments/S2/clean_articles.py` to create `Assignments/S2/data/cleaned_articles.json` without changing the raw download. It normalizes Unicode, removes the heading markers present in the plaintext extracts, and collapses duplicate whitespace. Wikipedia already provides plaintext with markup and references removed; bracketed text is preserved because it can contain useful linguistic information.

## Compare

Run `python Assignments/S2/evaluate_bpe.py` to evaluate the trained tokenizer. For each article, the script counts tokenizer IDs and unique whitespace-separated word forms; word forms are case-sensitive and punctuation is kept.

**Xᵢ = token count / unique word count**

### Results

| Language | Token count | Unique word count | Xᵢ |
| --- | ---: | ---: | ---: |
| English | 15,647 | 4,641 | 3.3715 |
| Hindi | 29,248 | 2,409 | 12.1411 |
| Telugu | 14,115 | 1,648 | 8.5649 |
| Tamil | 66,687 | 5,053 | 13.1975 |

**Ratio gap = 13.1975 - 3.3715 = 9.8260**  
**Score = 1000 / 9.8260 = 101.77**

## Train the shared tokenizer

Install the dependency with `python -m pip install -r requirements.txt`, then run `python Assignments/S2/train_bpe.py`. This trains one byte-level BPE tokenizer using all four cleaned articles. Its vocabulary has exactly 10,000 entries, including three reserved special tokens: `<|pad|>`, `<|bos|>`, and `<|eos|>`.

The complete tokenizer—including its vocabulary, merge rules, and configuration—is saved as `Assignments/S2/tokenizer.json`. The raw and cleaned article text remains in the ignored `data/` directory. The three special tokens are included in the 10,000-entry limit.

**Status:** Articles collected, normalized, tokenized, and evaluated. The score is a baseline for this single-article-per-language experiment.
# Session 2: A multilingual BPE tokenizer

## Experiment

Build a Byte Pair Encoding (BPE) tokenizer with one shared vocabulary of 10,000 tokens for the English, Hindi, Telugu, and Tamil versions of Wikipedia's India article.

## Run the full pipeline

Install the dependency with `python -m pip install -r requirements.txt`, then run:

```bash
python Assignments/S2/main.py
```

The pipeline downloads the four corresponding India articles if they are not already saved, cleans the text, trains the shared tokenizer, and evaluates it. It reuses saved articles by default so the results stay reproducible. To deliberately download the latest article versions and rerun every step, use `python Assignments/S2/main.py --refresh`.

The raw and cleaned article files are saved under `Assignments/S2/data/` and ignored by Git. Wikipedia's API provides plaintext with markup and references removed; cleaning additionally normalizes Unicode, removes plaintext heading markers, and collapses duplicate whitespace. Bracketed text is preserved.

## Evaluation

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

The tokenizer is byte-level BPE trained on all four cleaned articles. Its vocabulary contains exactly 10,000 entries, including `<|pad|>`, `<|bos|>`, and `<|eos|>`. The complete tokenizer—including its vocabulary, merge rules, and configuration—is saved as `Assignments/S2/tokenizer.json`.

**Status:** Articles collected, normalized, tokenized, and evaluated. The score is a baseline for this single-article-per-language experiment.
# Session 2: A multilingual BPE tokenizer

## Experiment

Build a Byte Pair Encoding (BPE) tokenizer with one shared vocabulary of 10,000 tokens for the English, Hindi, Telugu, and Tamil versions of Wikipedia's India article.

## Run the full pipeline

Install the dependency with `python -m pip install -r requirements.txt`, then run:

```bash
python Assignments/S2/main.py --experiment A
```

The pipeline downloads the four corresponding India articles if they are not already saved, cleans the text, trains the selected strategy, and evaluates it. It reuses saved articles by default so the results stay reproducible. To deliberately download the latest article versions and rerun every step, use `python Assignments/S2/main.py --experiment A --refresh`.

Run `python Assignments/S2/main.py --experiment B` to train with equal word counts. Experiment B keeps each article intact and deterministically samples additional word occurrences from shorter articles until each training text matches the longest article's word count. This avoids discarding vocabulary while balancing training volume. Both experiments use the same full cleaned articles for evaluation.

The raw and cleaned article files are saved under `Assignments/S2/data/` and ignored by Git. Wikipedia's API provides plaintext with markup and references removed; cleaning additionally normalizes Unicode, removes plaintext heading markers, and collapses duplicate whitespace. Bracketed text is preserved.

## Evaluation

For each language, split the cleaned text on whitespace, encode each distinct form once, and sum those token counts. Forms retain their original case and punctuation.

**Xᵢ = BPE tokens for unique forms / number of unique forms**

### Strategy experiments

Keep the same four cleaned articles for every evaluation. Only change how the shared tokenizer is trained; enforce exactly 10,000 vocabulary entries for every run.

| Experiment | Training strategy | Purpose | Status |
| --- | --- | --- | --- |
| A | Train on all four articles together | Baseline | Complete |
| B | Oversample shorter articles to the longest word count | Reduce article-length bias | Complete |
| C | Balance training text by character count | Compare script-sensitive effects | Planned |
| D | Tune language sampling weights | Reduce the ratio spread | Planned |

Do not assume balanced sampling produces balanced tokenization ratios. Select the strategy with the smallest measured ratio gap.

### Results: fixed evaluation set

For both strategies, evaluation always uses every unique whitespace-separated form from each of the same four cleaned articles. Only the tokenizer's training input changes.

#### Experiment A: all article text

| Language | Tokens for unique forms | Unique forms | Xᵢ |
| --- | ---: | ---: | ---: |
| English | 13,504 | 4,641 | 2.9097 |
| Hindi | 11,712 | 2,409 | 4.8618 |
| Telugu | 10,660 | 1,648 | 6.4684 |
| Tamil | 39,230 | 5,053 | 7.7637 |

**D = 7.7637 - 2.9097 = 4.8540**  
**Score = 1000 / 4.8540 = 206.02**

#### Experiment B: equal training word counts

Training used 12,112 word occurrences per language. English needed no additional samples; Hindi added 3,925, Telugu added 9,659, and Tamil added 1,889 sampled occurrences. Evaluation remained unchanged.

| Language | Tokens for unique forms | Unique forms | Xᵢ |
| --- | ---: | ---: | ---: |
| English | 13,492 | 4,641 | 2.9071 |
| Hindi | 11,701 | 2,409 | 4.8572 |
| Telugu | 10,650 | 1,648 | 6.4624 |
| Tamil | 39,235 | 5,053 | 7.7647 |

**D = 7.7647 - 2.9071 = 4.8576**  
**Score = 1000 / 4.8576 = 205.86**

Experiment B's score is slightly lower than A's (205.86 vs. 206.02), so equalizing word counts did not improve the score in this run.

Each tokenizer is byte-level BPE with exactly 10,000 entries, including `<|pad|>`, `<|bos|>`, and `<|eos|>`. Experiment artifacts are saved separately as `Assignments/S2/tokenizer_A.json` and `Assignments/S2/tokenizer_B.json`.

**Status:** Experiments A and B are complete. Experiments C and D remain planned.
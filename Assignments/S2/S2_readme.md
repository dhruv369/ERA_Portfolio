# Session 2: A multilingual BPE tokenizer

## Experiment

Build a Byte Pair Encoding (BPE) tokenizer with one shared vocabulary of 10,000 tokens for the English, Hindi, Telugu, and Tamil versions of Wikipedia's India article.

## Results widget

The static widget displays the latest verified Experiment D measurements and provides downloads for the full tokenizer, its complete ID-ordered vocabulary, and the metric/source data:

- [Open the live widget](https://tokenizerexperiment.netlify.app/)
- [Open the local widget source](./widget/index.html)
- [Download the complete Experiment D tokenizer](./widget/tokenizer_D.json)
- [Download the 10,000-token vocabulary](./widget/tokenizer-vocab.tsv)
- [Download the widget results data](./widget/results.json)

Regenerate the widget data after changing the tokenizer or evaluation inputs with `python Assignments/S2/generate_widget_data.py`. The widget is configured for Netlify with the repository-root `netlify.toml`.

## Run the full pipeline

Install the dependency with `python -m pip install -r requirements.txt`, then run:

```bash
python Assignments/S2/main.py --experiment A
```

The pipeline downloads the four corresponding India articles if they are not already saved, cleans the text, trains the selected strategy, and evaluates it. It reuses saved articles by default so the results stay reproducible. To deliberately download the latest article versions and rerun every step, use `python Assignments/S2/main.py --experiment A --refresh`.

Run `python Assignments/S2/main.py --experiment B` to train with equal word counts. Experiment B keeps each article intact and deterministically samples additional word occurrences from shorter articles until each training text matches the longest article's word count.

Run `python Assignments/S2/main.py --experiment C` to balance training by character count. Here, a character is one Unicode code point in Python's `len()`; each full article is kept and sampled whole words are added to shorter texts until they reach or slightly exceed the longest article's character count.

Run `python Assignments/S2/main.py --experiment D` to search language-specific integer weights. Each weight repeats that language's full article in the training input; the script evaluates a fixed candidate set and saves the tokenizer with the best measured score.

All experiments use the same full cleaned articles for evaluation.

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
| C | Oversample shorter articles to the longest character count | Compare script-sensitive effects | Complete |
| D | Search language-specific article repetition weights | Reduce the ratio spread | Complete |

Do not assume balanced sampling produces balanced tokenization ratios. Select the strategy with the smallest measured ratio gap.

### Results: fixed evaluation set

For all strategies, evaluation always uses every unique whitespace-separated form from each of the same four cleaned articles. Only the tokenizer's training input changes.

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

#### Experiment C: equal training character counts

Characters are Unicode code points. Training texts were balanced to approximately 93,085 characters each. English added 1,840 sampled words, Hindi added 8,694, Telugu added 9,088, and Tamil needed no additional words. Evaluation remained unchanged.

| Language | Tokens for unique forms | Unique forms | Xᵢ |
| --- | ---: | ---: | ---: |
| English | 13,516 | 4,641 | 2.9123 |
| Hindi | 11,698 | 2,409 | 4.8560 |
| Telugu | 10,650 | 1,648 | 6.4624 |
| Tamil | 39,240 | 5,053 | 7.7657 |

**D = 7.7657 - 2.9123 = 4.8534**  
**Score = 1000 / 4.8534 = 206.04**

Experiment C currently has the smallest spread and highest score, but its improvement over A is very small.

#### Experiment D: language-specific sampling weights

The search tested integer repetition weights in English:Hindi:Telugu:Tamil order. Each candidate used the same 10,000-entry vocabulary budget and the fixed evaluation set.

| Weights | D | Score |
| --- | ---: | ---: |
| 1:1:1:1 | 4.8540 | 206.02 |
| 1:1:1:2 | 4.8525 | 206.08 |
| 1:1:2:2 | 4.8503 | 206.17 |
| 1:2:2:2 | 4.8518 | 206.11 |
| 1:2:3:4 | 4.8462 | 206.35 |
| 1:2:3:6 | 4.8402 | 206.60 |
| **1:2:4:6** | **4.8380** | **206.70** |

Best-candidate results:

| Language | Tokens for unique forms | Unique forms | Xᵢ |
| --- | ---: | ---: | ---: |
| English | 13,579 | 4,641 | 2.9259 |
| Hindi | 11,684 | 2,409 | 4.8501 |
| Telugu | 10,649 | 1,648 | 6.4618 |
| Tamil | 39,231 | 5,053 | 7.7639 |

**D = 7.7639 - 2.9259 = 4.8380**  
**Score = 1000 / 4.8380 = 206.70**

The best weights in the current saved Experiment D artifact are 1:2:4:6. This improves slightly on Experiment C (206.04); the best weights are only the best among the seven candidates tested, not a global optimum.

Each tokenizer is byte-level BPE with exactly 10,000 entries, including `<|pad|>`, `<|bos|>`, and `<|eos|>`. Experiment artifacts are saved separately as `Assignments/S2/tokenizer_A.json`, `Assignments/S2/tokenizer_B.json`, `Assignments/S2/tokenizer_C.json`, and `Assignments/S2/tokenizer_D.json`.

**Status:** Experiments A, B, C, and D are complete. D's best result is from a limited candidate search.
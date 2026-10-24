# Session 2: A multilingual BPE tokenizer

## Experiment

Build a Byte Pair Encoding (BPE) tokenizer with one shared vocabulary of 10,000 tokens for the English, Hindi, Telugu, and Tamil versions of Wikipedia's India article.

## First step: prepare the articles

Run `python Assignments/S2/fetch_articles.py` to collect the corresponding India articles from Wikidata and Wikipedia. The script saves their plain text, page URLs, and revision IDs to `Assignments/S2/data/articles.json`. This downloaded corpus is local and ignored by Git.

Before comparing tokenization, decide on a consistent text-cleaning and word-counting rule for all four languages.

## Compare

For each language, compare the number of tokens produced with the number of words in its article:

**Tokenization ratio = token count / word count**

Compare the four ratios and aim to reduce the difference between the highest and lowest. The assignment score is **1000 / (highest ratio - lowest ratio)**, so a smaller gap gives a higher score.

**Status:** Articles collected; tokenizer not started.
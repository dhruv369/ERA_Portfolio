# Session 2: A multilingual BPE tokenizer

## Experiment

Build a Byte Pair Encoding (BPE) tokenizer with one shared vocabulary of 10,000 tokens for the English, Hindi, Telugu, and Tamil versions of Wikipedia's India article.

## Prepare the articles

Run `python Assignments/S2/fetch_articles.py` to collect the corresponding India articles from Wikidata and Wikipedia. The script saves their plain text, page URLs, and revision IDs to `Assignments/S2/data/articles.json`. This downloaded corpus is local and ignored by Git.

Run `python Assignments/S2/clean_articles.py` to create `Assignments/S2/data/cleaned_articles.json` without changing the raw download. It normalizes Unicode, removes the heading markers present in the plaintext extracts, and collapses duplicate whitespace. Wikipedia already provides plaintext with markup and references removed; bracketed text is preserved because it can contain useful linguistic information.

Next, define a consistent word-counting rule for all four languages before comparing tokenization.

## Compare

For each language, compare the number of tokens produced with the number of words in its article:

**Tokenization ratio = token count / word count**

Compare the four ratios and aim to reduce the difference between the highest and lowest. The assignment score is **1000 / (highest ratio - lowest ratio)**, so a smaller gap gives a higher score.

**Status:** Articles collected and normalized; tokenizer not started.
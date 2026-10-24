# Session 2: A multilingual BPE tokenizer

## Experiment

Build a Byte Pair Encoding (BPE) tokenizer with one shared vocabulary of 10,000 tokens for the English, Hindi, Telugu, and Tamil versions of Wikipedia's India article.

## Prepare the articles

Run `python Assignments/S2/fetch_articles.py` to collect the corresponding India articles from Wikidata and Wikipedia. The script saves their plain text, page URLs, and revision IDs to `Assignments/S2/data/articles.json`. This downloaded corpus is local and ignored by Git.

Run `python Assignments/S2/clean_articles.py` to create `Assignments/S2/data/cleaned_articles.json` without changing the raw download. It normalizes Unicode, removes the heading markers present in the plaintext extracts, and collapses duplicate whitespace. Wikipedia already provides plaintext with markup and references removed; bracketed text is preserved because it can contain useful linguistic information.

## Compare

Next, define one consistent word-counting rule for all four languages. Encode each cleaned article with the shared tokenizer and compare its token count with its word count:

**Tokenization ratio = token count / word count**

Compare the four ratios and aim to reduce the difference between the highest and lowest. The assignment score is **1000 / (highest ratio - lowest ratio)**, so a smaller gap gives a higher score.

## Train the shared tokenizer

Install the dependency with `python -m pip install -r requirements.txt`, then run `python Assignments/S2/train_bpe.py`. This trains one byte-level BPE tokenizer using all four cleaned articles. Its vocabulary has exactly 10,000 entries, including three reserved special tokens: `<|pad|>`, `<|bos|>`, and `<|eos|>`.

The complete tokenizer—including its vocabulary, merge rules, and configuration—is saved as `Assignments/S2/tokenizer.json`. The raw and cleaned article text remains in the ignored `data/` directory. The three special tokens are included in the 10,000-entry limit.

**Status:** Articles collected and normalized; shared 10,000-entry tokenizer trained. Tokenization-ratio comparison is next.
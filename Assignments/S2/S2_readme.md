# Session 2: A multilingual BPE tokenizer

## Experiment

Build a Byte Pair Encoding (BPE) tokenizer with one shared vocabulary of 10,000 tokens for the English, Hindi, Telugu, and Bengali versions of Wikipedia's India article.

## Compare

For each language, compare the number of tokens produced with the number of words in its article:

**Tokenization ratio = token count / word count**

Compare the four ratios and aim to reduce the difference between the highest and lowest. The assignment score is **1000 / (highest ratio - lowest ratio)**, so a smaller gap gives a higher score.

**Status:** Planning
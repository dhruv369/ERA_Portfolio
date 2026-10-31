# Session 4: An explainable cleaning funnel

## Deliverable and status

This assignment implements a bounded, reproducible preprocessing pipeline for **Samanantar**, plus a short interactive widget centered on a survival-funnel chart. The eight principal strategies are independently switchable in the widget's illustrative preview and in the command-line run. The browser does not claim to reclean the full corpus; change toggles and rerun the pipeline for measured results.

**A complete capped slice has now been processed.** It contains 9,999,995 input whitespace tokens across 458,258 sentence pairs, below the hard 10,000,000-token ceiling. The measured output is recorded in [`output/stats.json`](./output/stats.json); the cleaned records are in [`output/cleaned.jsonl`](./output/cleaned.jsonl). Token counts use maximal non-whitespace runs across both English and target text, not model-tokenizer tokens. Results below report measured counts, and decontamination is explicitly marked as not run.

**Deployment:** not deployed. The repository's existing Netlify publish configuration is intentionally unchanged.

## Dataset, source, and license

The selected source is the [AI4Bharat Samanantar dataset repository](https://huggingface.co/datasets/ai4bharat/samanantar), pinned in the pipeline to revision [`ead34c5b22a0354d350e7dfdd6aece7df5a48f37`](https://huggingface.co/datasets/ai4bharat/samanantar/tree/ead34c5b22a0354d350e7dfdd6aece7df5a48f37). Its Hugging Face dataset-card/API metadata identifies:

- **49.6 million English–Indic sentence pairs**, in 11 configs: Assamese (`as`), Bengali (`bn`), Gujarati (`gu`), Hindi (`hi`), Kannada (`kn`), Malayalam (`ml`), Marathi (`mr`), Odia (`or`), Punjabi (`pa`), Tamil (`ta`), and Telugu (`te`).
- Record fields `idx`, `src`, and `tgt`; the pipeline treats each pair as one document, with English in `src` and the config language in `tgt`.
- **Repository-card license label: CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0). This is the label declared by the dataset repository, not a legal review or proof that every underlying source has identical terms. Preserve attribution, review the [license terms](https://creativecommons.org/licenses/by-nc/4.0/), and investigate component-source restrictions before any use. Do not treat this non-commercially licensed corpus as cleared for commercial training.

The dataset card describes 49.6M pairs; the paper reports 49.7M, reflecting a source/presentation difference. This assignment follows the repository card for the downloaded repository and cites the paper for dataset background: Ramesh et al., [“Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages”](https://arxiv.org/abs/2104.05596), 2021. Session 3 discusses IndicTrans2 and 22 scheduled languages; Samanantar is relevant parallel data, but covers only these 11 Indic languages.

Metadata was checked against the dataset card and API on **2026-10-31**. The API reports repository revision and card metadata; the data has not been downloaded in this assignment. Network access to the metadata worked, but this is not evidence that a complete slice has been fetched or processed.

## Slice and token definition

The default run targets at most **10,000,000 whitespace tokens** across source **and** target text. A token here is exactly one maximal run of non-whitespace characters (`\S+`), not a SentencePiece, BPE, or model-tokenizer token. The hard limit applies to the raw input slice: processing stops before the next pair that would exceed the budget, so a run may end below the target. Requests over 10,000,000 are rejected. Output token counts are recounted after text transformations.

For reproducibility, the downloader pins the dataset revision and iterates language configs in the listed order, taking rows round-robin from streaming datasets until the token budget is reached. The output statistics record the revision, Python and `datasets` versions, token definition, configured languages, enabled/disabled strategies, actual slice counts, and each stage's input/output counts. A rerun needs network access and enough local disk for cleaned JSONL and a temporary SQLite dedup index. The index is removed after the run.

## Eight principal strategies

Stages run in this order; the order affects outcomes. `output/stats.json` records per-stage input/output document and whitespace-token counts, dropped documents, modifications, and findings. An edit does not count as a surviving-document loss.

| Strategy | Actual implementation and parameters | Status / limits |
| --- | --- | --- |
| **1. Boilerplate extraction** | Drops an entire pair if either side matches one of four explicit phrase rules (cookie prompts, subscription prompts, JavaScript requirement, or a standalone cookie/privacy-policy label). | Applied by default. It is intentionally narrow; common boilerplate may pass, and legitimate text can coincidentally match. |
| **2. Unicode normalization** | NFC on `src` and `tgt`; no NFKC, transliteration, whitespace collapse, or code formatting changes. U+200C ZWNJ and U+200D ZWJ are preserved. | Applied by default. Canonical composition can alter code-point counts but not whitespace. |
| **3. Ghost / special-tag handling** | Removes matched visible spellings such as `[USER]`, `[ASSISTANT]`, `<|endoftext|>`, and `<extra_id_12>`; records occurrences and changed pairs. | Applied by default. This is a declared pattern list, not a universal tokenizer-control-token detector. |
| **4. Language ID** | Assigns source English and target language from the selected Samanantar config. A Unicode-script-range heuristic flags target text with under 25% expected-script letters; it does not drop it. | Metadata-based labeling, **not an independent language-ID classifier**. A classifier is skipped because no multilingual detector/model was selected and validated across these scripts, code-switching, and transliteration. Script flags are only audit hints; transliteration and mixed script may be misflagged. |
| **5. Heuristic quality filtering** | Drops pairs if either side is under 5 chars or over 2,000 chars; has a whitespace token over 100 chars; has under 15% letters when at least 20 chars; or contains a character repeated 10+ times. Also drops a pair if the longer side has over 8× the shorter side's whitespace-token count. Records each rule firing. | Applied by default; no learned quality classifier. Short utterances, names, numeric/code-heavy text, transliteration, and very different but valid translations may be over-filtered. Inspect samples per language and compare with filters disabled. |
| **6. Decontamination** | If `--holdout-file` is supplied, each non-empty line of at least 40 chars is case-folded and whitespace-normalized; a pair is dropped when either side contains/is contained by a holdout line. | Otherwise explicitly `not_run_no_holdout_file`. No benchmark/reference list is bundled. This exact substring check is not a comprehensive n-gram, paraphrase, or benchmark contamination audit. It runs before PII replacement and dedup so the scrubbing stage cannot hide a literal holdout match, and removed rows never enter the dedup index. |
| **7. PII scrubbing** | Regex patterns find email-like strings and phone-like digit sequences; replacements are `[EMAIL]` and `[PHONE]`. Findings count pattern matches, not unique people or confirmed PII. | Applied by default. Regexes have false positives and miss other identifiers, addresses, context-dependent PII, and obfuscated values. This is not a privacy guarantee. |
| **8. MinHash + LSH deduplication** | Pair text is tokenized into case-folded five-word shingles (or all words for shorter pairs); each shingle is hashed with 64-bit BLAKE2b, then 128 deterministic affine MinHash permutations over `2^61−1`; 32 bands × 4 rows retrieve candidates, followed by exact shingle Jaccard at **0.85**. First occurrence wins. SQLite stores candidate buckets and survivor shingles. | Applied by default after decontamination and PII replacement. The approximate LSH candidate search can miss duplicates; near-duplicate semantics are not captured by the score. The reported rate is removed candidate-confirmed rows divided by rows reaching dedup; it is not a recall-corrected global duplicate rate. |

### Concrete examples and over-filtering checks

- `Cafe\u0301 guide` becomes `Café guide` under NFC. Whitespace stays byte-for-byte unchanged; code spacing is not collapsed.
- `नमस्ते\u200d` keeps its ZWJ, and Devanagari ZWNJ is likewise not stripped.
- `Good morning [USER]` loses the matched visible tag and increments the ghost-tag occurrence counter.
- `Contact learner@example.org at +1 555 010 2345` is scrubbed to placeholders; this shows pattern behavior, not an observed source example or corpus PII rate.
- A pair containing `Subscribe to continue reading` is removed by a boilerplate phrase match. A real subscription discussion might be incorrectly removed; review matched examples.
- One- or two-word source/target sentences may fail the 5-character or length-ratio heuristic. Compare counts and native-speaker samples with `quality` disabled before deciding those rules are suitable.

The widget's eight handcrafted examples are synthetic demonstrations. They are never included in the dataset result counts.

## Reproduce locally

From the repository root, install the one assignment-specific dependency and run tests:

```sh
python -m pip install -r Assignments/S4/requirements.txt
python -m unittest discover -s Assignments/S4 -p 'test_*.py'
```

Fetch/process the default bounded slice:

```sh
python Assignments/S4/pipeline.py \
  --target-tokens 10000000 \
  --output-dir Assignments/S4/output
```

Run a comparison with selected stages disabled (stages not listed remain enabled):

```sh
python Assignments/S4/pipeline.py \
  --target-tokens 10000000 \
  --disable quality,minhash_lsh \
  --output-dir Assignments/S4/output-no-quality-dedup
```

For a local, explicitly chosen holdout list, pass `--holdout-file path/to/holdout-lines.txt`. Do not add private or sensitive material to that file. Output is `cleaned.jsonl` and `stats.json` inside the chosen output directory. The pipeline does not copy corpus excerpts into its report; examples in the widget and this README are synthetic.

Serve the repository root to allow the widget to load measured stats (if generated):

```sh
python -m http.server 8000
```

Open `http://localhost:8000/Assignments/S4/widget/`. If `output/stats.json` is absent, the funnel shows only the eight-item illustrative demonstration. Browser toggles rerun that demonstration; they do not change the already measured pipeline output. The command-line `--disable` option creates a new, separately measured run.

## Measured results

The run was generated at **2026-10-31 22:59:49 UTC** using Python 3.10.19 and `datasets` 4.8.5, against the pinned revision above. The input budget is hard-capped at 10,000,000 whitespace tokens; 5 tokens remained unused because the next pair would have exceeded it.

### Survival funnel

Each row shows sentence pairs and whitespace tokens after that stage. Text-only edits can change tokens without dropping documents. The decontamination row is a carry-through because no holdout file was supplied.

| Funnel stage | Pairs remaining | Whitespace tokens remaining | Measured change / status |
| --- | ---: | ---: | --- |
| Input | 458,258 | 9,999,995 | 11 configs streamed round-robin |
| Boilerplate extraction | 458,256 | 9,999,987 | 2 pairs removed |
| NFC normalization | 458,256 | 9,999,987 | 68,428 pairs modified |
| Ghost / special tags | 458,256 | 9,999,987 | 0 occurrences found |
| Language metadata / script audit | 458,256 | 9,999,987 | 133 script-mismatch flags; no drops |
| Heuristic quality filter | 458,001 | 9,985,841 | 255 pairs removed; 268 rule firings |
| Decontamination | 458,001 | 9,985,841 | Not run: no holdout file |
| PII scrubbing | 458,001 | 9,984,503 | 940 pairs modified; 1,685 pattern matches |
| MinHash + LSH deduplication | **457,922** | **9,982,074** | 79 near-duplicate candidates removed |

### Final language split

Language is taken from each dataset config, not independently classified. These are output counts after all enabled stages; output token totals include the English source and the indicated target language.

| Config | Output pairs | Output whitespace tokens | Script-mismatch flags |
| --- | ---: | ---: | ---: |
| Assamese (`as`) | 41,563 | 933,287 | 31 |
| Bengali (`bn`) | 41,641 | 901,946 | 11 |
| Gujarati (`gu`) | 41,646 | 822,384 | 10 |
| Hindi (`hi`) | 41,643 | 1,438,024 | 14 |
| Kannada (`kn`) | 41,636 | 685,097 | 4 |
| Malayalam (`ml`) | 41,634 | 712,909 | 9 |
| Marathi (`mr`) | 41,645 | 865,254 | 2 |
| Odia (`or`) | 41,645 | 900,927 | 12 |
| Punjabi (`pa`) | 41,636 | 1,204,966 | 13 |
| Tamil (`ta`) | 41,586 | 816,759 | 11 |
| Telugu (`te`) | 41,647 | 700,521 | 16 |

The 133 script flags are audit hints only, not verified language errors. The slice was streamed approximately evenly by pair count, not balanced by token count; language token totals therefore differ.

### Other measured findings and caveats

- **Deduplication:** 79 of 458,001 pairs reaching dedup were removed (**0.0172%**). The 0.85 Jaccard threshold uses 128 MinHash permutations and 32 bands × 4 rows over five-word pair shingles. LSH candidate retrieval is approximate, so this is not a recall-corrected global duplicate rate.
- **PII patterns:** 17 email-like and 1,668 phone-like matches were replaced across 940 pairs. These are regex match counts, not confirmed unique people; false positives and missed identifiers are possible.
- **Quality rules:** the 268 rule firings included 202 extreme source/target length-ratio flags, 49 low-letter-ratio flags, 8 repeated-character flags, 6 over-2,000-character flags, and 3 extreme-token flags. Short valid translations, names, code/numeric content, transliteration, or code-mixed content may be over-filtered.
- **Ghost tags:** none of the configured visible special-token spellings were found in this slice.
- **Decontamination:** not performed. There was no benchmark holdout list, so no benchmark-contamination claim can be made.
- **Unicode:** NFC changed 68,428 pairs; ZWJ/ZWNJ and whitespace were preserved. Canonical normalization does not guarantee all text is semantically or linguistically correct.
- **Examples in widget:** handcrafted and illustrative, not sampled from the corpus. The widget loads the measured funnel from `output/stats.json` when served from the repository root.
- **License and privacy:** the dataset card declares CC-BY-NC-4.0; component-source terms need review. Regex scrubbing is not a privacy guarantee.
- **Deployment:** not deployed; the existing Netlify publish configuration was not changed.

The summary numbers above are cross-checked against the saved statistics: the per-language output pair counts sum to 457,922, matching both the reported output and the number of records in `cleaned.jsonl`. Review source samples and compare with stages disabled before treating a higher survival rate as better data quality.

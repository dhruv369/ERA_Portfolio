# Designing a 40B India-first model: data, evaluation, and a fair test

*A research-backed design proposal—not a claim that this model has been trained.*

The challenge is not simply to build a 40-billion-parameter model. It is to code, use tools reliably, reason across science and mathematics, and serve India’s linguistic diversity. “India-first” should mean grounding the model in India’s languages and varied lived realities—not promoting one political viewpoint.

I would treat Gemma 4 as a benchmark to test against, not a result a 40B model can claim in advance. Gemma 4 includes a 31B dense model and a 26B mixture-of-experts model with about 4B active parameters per token, as well as multimodal capabilities. A 40B dense model is not automatically better, cheaper, or equivalent. Comparisons must use matched tasks, prompts, tools, context lengths, and inference budgets.

## 1. Build the data mix around capabilities

For pre-training, I would plan an initial **800-billion-token run** and use these sampling weights:

| Primary sampling bucket | Share | What it contributes |
| --- | ---: | --- |
| High-quality global and English text | 40% | Broad knowledge, writing, and technical material |
| Indic-language text | 25% | Native-script, code-switched, and Romanized content across India’s languages |
| Code and software engineering | 15% | Source code, tests, documentation, issue discussions, and repository changes |
| Mathematics and science | 10% | Textbooks, worked examples, papers, and verified structured problems |
| India-focused public-interest material | 5% | Rights-cleared civic, education, agriculture, health, and legal information |
| Other languages and structured data | 5% | Broader multilingual coverage and useful structured formats |

These are **sampling hypotheses, not measured optima**. Assign each example one primary bucket to avoid double counting, while retaining language and domain tags. Set exposure floors for each language and adjust from evaluations, not raw web volume.

The 22 scheduled languages—Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Meitei (Manipuri), Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, and Urdu—are a starting scope, not the limit of India’s linguistic diversity. Include regional varieties, native scripts, transliteration, and code-switching. IndicTrans2 demonstrates the value of India-relevant parallel data and evaluation across these languages.

Post-training should teach how to respond, not just predict text. Curate multilingual instruction examples for explanations, translation, coding, debugging, repository changes, and structured tool use. Agent traces should include tool selection, checking results, error recovery, and stopping when done. Use native speakers and domain experts for preference comparisons.

Use reinforcement learning where rewards can be checked: code tests, verifiable math, or tool tasks with explicit success conditions. For open-ended or culturally sensitive questions, human review is safer than optimizing an unverified reward.

## 2. Clean for quality without erasing diversity

Every source should have a record of where it came from, when it was collected, its license or terms, language and script, transformations, and exclusions. Publicly accessible does not automatically mean legally reusable; rights review should happen before training.

Identify language and script, remove spam and boilerplate, normalize corrupt text, and deduplicate documents and passages. Check for benchmark leakage and sensitive personal information. For code, preserve repository context and license metadata.

Aggressive normalization can erase dialects, transliteration, and mixed-language speech. Audit samples by language after each filter, and involve native speakers. Track data quality as well as volume so English cannot hide a weak language pipeline.

## 3. Evaluate the claims directly

Report results **by language, script, and task**, not one overall score.

- **Indic languages:** Use IndicTrans2’s India-origin, 22-language resources for translation, plus held-out native-speaker tests for instruction following, factual answers, and code-switching. Review adequacy, fluency, and factuality.
- **Coding:** Use LiveCodeBench’s time-stamped problems and SWE-bench Verified for repository issues. Report the harness, test execution, time limit, and resolution rate.
- **Agents:** Test interactive task completion and tool correctness. AgentBench and OSWorld are starting points; add India-relevant workflows and report failures and recovery.
- **Math and science:** Combine GSM8K, MATH, GPQA, and MMLU-Pro with held-out expert questions. No single benchmark proves broad reasoning ability.

Keep test sets and solutions out of training, use release-date cutoffs and private tests, and report contamination checks. For Gemma 4 comparisons, match tasks, tools, context, and inference budgets; report mixed results too.

## 4. Set tokenizer targets, then measure them

Test a **128,000-token SentencePiece unigram vocabulary** against 64K and 96K candidates, using balanced Indic, English, code, STEM, and code-switched samples. Select by fertility and downstream quality; larger vocabularies cost parameters and can make rare pieces sparse.

Define fertility before comparing tokenizers. These are **proposed acceptance targets**, not universal standards:

| Content | Measurement and target |
| --- | --- |
| Each of the 22 scheduled languages | On meaning-matched text, median tokens per word no more than **1.5×** the English equivalent; investigate any language with a **95th-percentile ratio above 2×**. Measure native script, transliteration, and code-switching separately. |
| Code | On held-out repositories, keep tokens per byte within **10% of the best** of the candidate tokenizers; check that identifier and punctuation splitting does not damage code-task results. |
| Mathematics and science | On held-out formulas and technical passages, stay within **10% of the lowest tokens per character** among candidates, while preserving benchmark performance. |
| Agent traces and tool calls | Keep each dedicated control token atomic; compare tokens per equivalent successful trace and require no more than **10% overhead** versus the most efficient candidate. |

Research has found substantial tokenization disparities across languages. My ratios are engineering goals to test, not published norms.

At model width 8,192, a 128K input embedding has about **1.05 billion parameters** (128,000 × 8,192), before an untied output embedding. That is a meaningful part of a 40B parameter budget, so tokenizer efficiency must be balanced against model capacity.

## 5. Check whether 40B is worth the compute

A 40B dense model needs about **80 GB for BF16 weights alone**, before optimizer states, activations, and serving cache. Chinchilla’s compute-optimal result suggests roughly 20 training tokens per parameter: **800 billion tokens** for 40B, or approximately **1.9 × 10²³ FLOPs** using `6 × parameters × tokens`, before overhead. Treat this as a baseline, not a guarantee. Scale toward multiple trillions only if pilot runs show the data and compute are worthwhile.

Before committing to the full run, I would train smaller scaling-law pilots, validate the data mix and tokenizer, and compare quality per unit of training and inference compute. The key question is not whether 40B sounds large, but whether this design delivers reliable improvements across India’s languages and the targeted coding and agent tasks at a cost people can use.

## Sources

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) and [technical report](https://arxiv.org/abs/2607.02770)
- [IndicTrans2: 22 scheduled Indian languages and India-relevant evaluation](https://arxiv.org/abs/2305.16307)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)
- [SentencePiece](https://arxiv.org/abs/1808.06226)
- [LiveCodeBench](https://livecodebench.github.io/) and [SWE-bench](https://www.swebench.com/)
- [AgentBench](https://arxiv.org/abs/2308.03688) and [OSWorld](https://arxiv.org/abs/2404.07972)
- [GSM8K](https://arxiv.org/abs/2110.14168), [MATH](https://arxiv.org/abs/2103.03874), [GPQA](https://arxiv.org/abs/2311.12022), and [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [Digital Personal Data Protection Act, 2023](https://www.indiacode.nic.in/handle/123456789/22067)

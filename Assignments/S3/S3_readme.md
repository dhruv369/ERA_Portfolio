# Session 3: Researching an India-first 40B model

## Assignment

Develop a research-backed plan for training a 40B-parameter model that aims to match Gemma 4 while excelling at coding, agentic work, and Indic languages, and reflecting an India-first perspective.

This is a design exercise, not a claim that a model has been trained. The report will explain the choices, trade-offs, and evidence behind a practical training and evaluation plan.

## Questions the report will answer

- **Data:** What to collect for pre-training, post-training, and reinforcement learning or alignment, and why each source supports the target capabilities.
- **Quality and cleaning:** How to filter, deduplicate, balance, document, and govern data without erasing useful linguistic or cultural variation.
- **Evaluation:** How to measure language quality, coding, agentic task completion, science, math, and India-relevant knowledge, including held-out and contamination-resistant tests.
- **Tokenization:** Which languages and domains to prioritize, what fertility targets to use, and how those targets inform the vocabulary size.
- **Trade-offs:** What a 40B-parameter budget makes feasible, what remains uncertain, and how to validate the plan before training.

## Status

The research-backed Medium draft is ready in [S3_medium_post.md](./S3_medium_post.md). Its figures and recommendations are design targets, not experimental results. Review the assumptions and sources before publishing; the draft has not been uploaded to Netlify or Medium.
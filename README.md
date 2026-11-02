# AI & Deep Learning Learning Portfolio

This repository is where I experiment with AI, deep learning, and, as I progress, large language models (LLMs).
I learn by building small projects that make the ideas visible and testable. Each session has its own README with the experiment details, so this page can stay a simple index.

---

## Featured project

[Session 3: Researching an India-first 40B model](./Assignments/S3/S3_readme.md) includes a [Medium-ready research draft](./Assignments/S3/S3_medium_post.md) on data, cleaning, evaluation, Indic-language tokenization, and training trade-offs.

## All assignments

| Assignment | Project | Status |
| --- | --- | --- |
| 01 | [Deep learning fundamentals](./Assignments/S1/S1_readme.md) | Done |
| 02 | [A multilingual BPE tokenizer](./Assignments/S2/S2_readme.md) | In progress |
| 03 | [Researching an India-first 40B model](./Assignments/S3/S3_readme.md) | Draft ready |
| 04 | [Cleaning a text dataset](./Assignments/S4/S4_readme.md) | Planned |
| 05 | [V5 data mixture and curriculum](./Assignments/S5/S5_readme.md) | Draft ready |
| 10 | [Making a small model tell the truth](./Assignments/S10/README.md) | In progress |
| 11 | [Understanding Adam and learning-rate schedules](./Assignments/S11/README.md) | Done |
| 12 | [Simulating ZeRO across 32 virtual GPUs](./Assignments/S12/README.md) | Done |
| 13 | [Training a 20M-parameter language model](./Assignments/S13/README.md) | In progress |
| 14 | [Turning a linear model into an MoE](./Assignments/S14/README.md) | Planned |
| 15 | [Capstone project](./Assignments/S15/README.md) | Planned |

## Technical skills

Tools and concepts used so far:

- **Languages and web:** Python, HTML, CSS, JavaScript
- **Machine learning:** TensorFlow.js, PyTorch, NumPy, Matplotlib, Hugging Face Tokenizers
- **Concepts:** neural networks, ReLU activations, model depth, embeddings, next-token prediction, train/test generalization, byte-level BPE

## Learning journey

- **Session 1:** Built experiments around core deep learning ideas. See the [session notes](./Assignments/S1/S1_readme.md) for what was tried and learned.
- **Session 2:** Compared baseline, word-balanced, character-balanced, and weighted training for a shared 10,000-entry BPE tokenizer. The best tested weights scored marginally highest. See the [session notes](./Assignments/S2/S2_readme.md).
- **Session 3:** Drafted a research-backed data, training, evaluation, and tokenization plan for an India-first 40B model. See the [research brief](./Assignments/S3/S3_readme.md) and [Medium-ready article](./Assignments/S3/S3_medium_post.md).
- **Session 4:** Planning an interactive, stage-by-stage text-cleaning widget for a 10–100M dataset, with a focus on preserving Indic text. See the [session brief](./Assignments/S4/S4_readme.md).
- **Session 5:** Specified a proposed V5 data mixture, Indic allocation, curriculum, and 1B/3B proxy experiments, with unverified inventory and thresholds marked TBD. See the [session brief](./Assignments/S5/S5_readme.md).
- **Session 10:** Building a small model and checking gradients, accumulation, training signals, utilization, and floating-point representations. See the [session brief](./Assignments/S10/README.md).
- **Session 11:** Completed hand-checked Adam updates, bias-correction and warmup experiments, a tuned cosine/WSD comparison, and multi-seed learning-rate sweeps at widths 256, 512, and 1,024. The width-4,096 learning-rate recommendation is explicitly low-confidence. See the [session report and reproducible evidence](./Assignments/S11/README.md).
- **Session 12:** Completed a 32-rank CPU simulation of ZeRO-0 through ZeRO-3, including a trained MLP, sharded Adam updates, and explicit memory/FLOP/communication estimates. The report distinguishes simulated ranks from real multi-GPU measurements. See the [session report and notebook](./Assignments/S12/README.md).
- **Session 13:** Training a 20M-parameter language model and comparing standard and reversible runs. See the [session brief](./Assignments/S13/README.md).
- **Session 14:** Converting a trained linear model into an MoE and verifying that training continues to reduce loss. See the [session brief](./Assignments/S14/README.md).
- **Session 15:** Capstone project details to be added. See the [capstone README](./Assignments/S15/README.md).

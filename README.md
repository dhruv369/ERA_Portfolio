# AI & Deep Learning Learning Portfolio

This repository is where I experiment with AI, deep learning, and, as I progress, large language models (LLMs).
I learn by building small projects that make the ideas visible and testable. Each session has its own README with the experiment details, so this page can stay a simple index.

---

## Featured project

[Session 3: Researching an India-first 40B model](./Assignments/S3/S3_readme.md) is a research brief on data, cleaning, evaluation, Indic-language tokenization, and training trade-offs.

## All assignments

| Assignment | Project | Status |
| --- | --- | --- |
| 01 | [Deep learning fundamentals](./Assignments/S1/S1_readme.md) | Done |
| 02 | [A multilingual BPE tokenizer](./Assignments/S2/S2_readme.md) | In progress |
| 03 | [Researching an India-first 40B model](./Assignments/S3/S3_readme.md) | Research in progress |
| 10 | [Making a small model tell the truth](./Assignments/S10/README.md) | In progress |
| 11 | [Understanding Adam and learning-rate schedules](./Assignments/S11/README.md) | In progress |
| 12 | [Simulating ZeRO across 32 virtual GPUs](./Assignments/S12/README.md) | In progress |
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
- **Session 3:** Researching a data, training, evaluation, and tokenization plan for an India-first 40B model. See the [research brief](./Assignments/S3/S3_readme.md).
- **Session 10:** Building a small model and checking gradients, accumulation, training signals, utilization, and floating-point representations. See the [session brief](./Assignments/S10/README.md).
- **Session 11:** Reproducing Adam updates and evaluating bias correction, warmup, learning-rate schedules, and scaling across model widths. See the [session brief](./Assignments/S11/README.md).
- **Session 12:** Simulating 32 virtual devices and comparing the memory and computation tradeoffs of ZeRO stages 1–3. See the [session brief](./Assignments/S12/README.md).
- **Session 13:** Training a 20M-parameter language model and comparing standard and reversible runs. See the [session brief](./Assignments/S13/README.md).
- **Session 14:** Converting a trained linear model into an MoE and verifying that training continues to reduce loss. See the [session brief](./Assignments/S14/README.md).
- **Session 15:** Capstone project details to be added. See the [capstone README](./Assignments/S15/README.md).

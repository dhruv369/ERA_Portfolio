# Session 13: Training a 20M-parameter language model

## Assignment

Train a language model with approximately 20 million parameters for 50 million tokens on Google Colab or another available environment. Compare a standard run with reversible-model variants, then test the best-performing reversible variant at the maximum batch size that fits your environment.

## Training runs

### 1. Standard baseline

Choose and document a batch size that runs reliably on the selected hardware. Train the model for a 50-million-token budget and record the configuration and results.

### 2. Reversible variant at the fixed batch size

Train again using reversibility with the same batch size and a comparable token budget. Try and report the reversible variant that works for you (for example, midpoint or Euler). Explain which variant you selected and why.

### 3. Reversible variant at maximum batch size

Train again with the selected reversible variant, increasing the batch size to the maximum that fits the available hardware. Record the largest successful batch size and any constraints or changes required to reach it.

## Results to report

For every run, report:

- Model parameter count, architecture, tokenizer, data, and tokens processed.
- Hardware and software environment, sequence length, batch size, and optimizer configuration.
- Final loss and training speed in tokens per second.
- Peak memory use and how it was measured.
- Whether the run completed its intended token budget, and any failures or adjustments.
- Other findings, including quality, stability, and tradeoffs observed between the standard and reversible runs.

Keep the comparisons reproducible. State any settings that differ between runs, and avoid attributing a result to reversibility or batch size if other changes could explain it.

## Submission

- Jupyter notebook(s) in the GitHub repository containing the training code, measurements, and analysis.
- A detailed README documenting the setup, all three runs, results, and conclusions.
- The GitHub repository link for review.

## Status

The standard baseline notebook and training script are prepared. The baseline uses TinyStories, the GPT-2 tokenizer (without pretrained model weights), a 19.86M-parameter decoder-only Transformer, batch size 16, and sequence length 256. A short GPU/model smoke-test cell runs before any dataset download or training. The training run is designed to process exactly 50 million tokens, with resumable Google Drive checkpoints and measurements for validation loss, throughput, and peak CUDA memory.

The notebook has not yet been run on a Colab GPU. Results are intentionally left unreported until the run completes. The reversible comparisons and maximum-batch run remain to be implemented after the baseline is measured.

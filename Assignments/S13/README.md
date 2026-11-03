# Session 13: Training a 20M-parameter language model

## Assignment

Train a language model with approximately 20 million parameters for 50 million tokens on Google Colab or another available environment. Compare a standard run with reversible-model variants, then test the best-performing reversible variant at the maximum batch size that fits your environment.

## Training runs

### 1. Standard baseline

`session13_baseline.py` trains a decoder-only Transformer from random initialization. It uses 3 blocks, width 320, 4 attention heads, tied input/output token embeddings, and has 19,863,680 parameters with the GPT-2 vocabulary. The fixed baseline batch size is 16 sequences of 256 tokens (4,096 tokens per full step).

### 2. Reversible variant at the fixed batch size

`session13_reversible.py` uses a 10-block additive reversible-coupling Transformer with 19,257,600 parameters. For each block, `y1 = x1 + F(x2)` and `y2 = x2 + G(y1)`. The backward pass reconstructs `x2 = y2 - G(y1)` and `x1 = y1 - F(x2)`, then recomputes block activations. This is the exact additive-coupling variant, not an Euler or midpoint ODE integrator. Its custom backward was checked against ordinary autograd, including CPU bfloat16 autocast; the coupling inverse was also numerically checked.

The fixed-batch comparison uses batch size 16 and sequence length 256, matching the baseline.

### 3. Reversible variant at maximum batch size

The notebook invokes the reversible script's `--find-max-batch` option. It probes batch sizes with a full forward/backward/optimizer step and searches for the largest batch that fits the selected GPU. It then trains a separate 50-million-token run at that batch size.

## Colab workflow

Open `session13.ipynb` in Colab with a GPU runtime and run the cells in order. The notebook installs the data/tokenizer packages, mounts Google Drive, stores the Hugging Face cache, tokenized TinyStories data, run checkpoints, and JSON results on Drive, and starts the three runs separately. The token cache is shared, so the TinyStories stream is tokenized only once. Checkpoints are written every 500 steps and runs resume after a disconnect when relaunched with the same configuration. Push the notebook and scripts to GitHub before using the notebook's repository-clone cell.

## Results to report

For every run, report:

- Model parameter count, architecture, tokenizer, data, and tokens processed.
- Hardware and software environment, sequence length, batch size, and optimizer configuration.
- Final loss and training speed in tokens per second.
- Peak memory use and how it was measured.
- Whether the run completed its intended token budget, and any failures or adjustments.
- Other findings, including quality, stability, and tradeoffs observed between the standard and reversible runs.

Keep the comparisons reproducible. State any settings that differ between runs, and avoid attributing a result to reversibility or batch size if other changes could explain it.

## Current status and results

| Run | Status | Batch size | Training-token budget |
| --- | --- | ---: | ---: |
| Standard baseline | Ready; Colab GPU run pending | 16 | 50,000,000 |
| Reversible additive coupling | Ready; Colab GPU run pending | 16 | 50,000,000 |
| Reversible additive coupling, maximum batch | Ready; Colab GPU run pending | GPU-probed | 50,000,000 |

No final loss, speed, or GPU-memory values are reported yet because the CUDA training runs have not been executed. Each completed script writes a JSON result file containing those measurements; use the notebook's final cell to inspect the files and then copy the measured results here.

## Submission

- Jupyter notebook(s) in the GitHub repository containing the training code, measurements, and analysis.
- A detailed README documenting the setup, all three runs, results, and conclusions.
- The GitHub repository link for review.

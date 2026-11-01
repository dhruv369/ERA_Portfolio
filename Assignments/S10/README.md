# Session 10: Making a small model tell the truth

## Assignment

Build a small model and a real training loop, then use measurements to understand what the model and training process are doing. This session emphasizes checking the details rather than trusting a plausible-looking loss curve.

## Experiments and evidence

### 1. Inspect tensor shapes

Print every tensor shape used in a training step. For each shape, explain what every dimension represents. Keep these explanations next to the corresponding output in the notebook.

### 2. Verify a gradient numerically

Choose a weight and nudge it by a small amount. Measure the resulting change in loss and compare the finite-difference estimate with the gradient reported by `backward()`. Show the values to several decimal places and explain any meaningful disagreement.

### 3. Demonstrate incorrect gradient accumulation

Use micro-batches of different lengths to compare:

- The incorrect average of per-micro-batch average losses.
- The correctly weighted loss/gradient across all examples or tokens.

Plot both curves together so the difference is visible. Explain why averaging averages gives the wrong weighting when micro-batch lengths differ.

### 4. Relate gradient norms to loss

Log the gradient norm at every training step. Identify and explain at least one step where the gradient norm changed before the loss did.

### 5. Estimate model FLOPs utilization

Compute and report your own MFU (model FLOPs utilization). Show the assumptions and quantities used in the estimate, report the result honestly, and explain what you believe is responsible for the distance from 40%.

### 6. Compare floating-point representations

Write out the bit representation of `0.1` in fp32, bf16, and fp8 E4M3. Explain how each format encodes the value, then state which format you would train in and why.

## Submission

- A Jupyter notebook named `session10.ipynb` containing the model, training loop, experiments, outputs, plots, calculations, and explanations.
- This README, updated with the final results and conclusions after the experiments are complete.
- The GitHub repository containing both files.

## Status

All six experiment areas are implemented in `session10.ipynb`. It contains a 108,320-parameter character-level causal Transformer, a self-contained toy corpus, training loop, tensor-shape trace, experiments, plots, and saved outputs. In the recorded 500-update run, mean training loss fell from `3.5725` to `0.0807` and validation loss from `3.5701` to `0.0788`. Because the corpus is repeated, this is a pipeline check, not evidence of generalization.

The shape trace prints 55 major tensor shapes for a forward training step, including embeddings, Q/K/V projections and heads, attention scores and weights, both Transformer blocks, feed-forward activations, flattened cross-entropy inputs, and scalar loss. Dimension labels define batch `B`, sequence length `T`, embedding width `C`, head count `H`, per-head width `Dh`, and vocabulary size `V`.

The numerical gradient check compares `lm_head.weight[0, 0]` by central difference (`epsilon = 0.01`) on a fixed batch. Backprop gave `-0.00966216`; the finite-difference estimate was `-0.00966787`, an absolute difference of `5.71e-6` (about `0.059%` relative error). This small discrepancy is consistent with finite-precision arithmetic and the finite perturbation size.

For accumulation, three micro-batches of eight sequences have lengths 8, 24, and 64 (64, 192, and 512 target tokens). The unweighted average of their mean losses was `3.573558`, versus `3.600885` when weighted by token count. Their correct token shares are `8.3%`, `25.0%`, and `66.7%`, rather than one third each. The full-model gradient norms were `0.883425` for incorrect accumulation and `0.800863` for token-weighted accumulation; the gradients differ by `39.91%` of the correct gradient norm. The notebook plots both output-head bias gradient vectors and their objective weights.

The training loop logs a global gradient L2 norm at all 500 updates and evaluates the same fixed probe batch after every update. The norm falls from `0.782626` to `0.274952` overall. At update 212, the 10-update mean gradient norm changed by `3.47%`; the smoothed probe loss changed by only `0.000003` at that point, then by `0.009562` over the following five updates. This is an observed lag in this run, not evidence that the gradient norm alone caused the later loss change.

For the MFU estimate, the run processes 2,048 tokens/update. The approximate forward-plus-backward count is `1.233 GFLOPs` for dense linear weights plus `0.201 GFLOPs` for attention, or `1.434 GFLOPs/update`. Measured forward/backward/update time averaged `4.971 ms`, giving an estimated `288.56 GFLOP/s`. With an explicitly assumed `4 TFLOP/s` peak, apparent MFU is `7.21%` (the same measurement gives `9.62%` at an assumed 3 TFLOP/s and `5.77%` at 5 TFLOP/s). The Apple M4 Pro / PyTorch MPS setup does not expose a validated peak FP32 throughput, so these percentages are illustrative scenarios, not a certified hardware MFU. At the 4 TFLOP/s assumption, 40% would require about `1.6 TFLOP/s`, around 5.5 times the measured estimate. The tiny model and short sequences, eager Python/MPS dispatch, and non-fused attention are likely reasons utilization is low; the unknown peak means the exact distance from 40% cannot be established here.

The observed bits for decimal `0.1` are:

| Format | Sign / exponent / fraction bits | Decoded value |
| --- | --- | ---: |
| fp32 | `0 / 01111011 / 10011001100110011001101` | `0.100000001490` |
| bf16 | `0 / 01111011 / 1001101` | `0.100097656250` |
| fp8 E4M3 | `0 / 0011 / 101` | `0.101562500000` |

None represents `0.1` exactly. For mixed-precision training, bf16 is the preferred activation/matmul format when supported, with optimizer states and sensitive accumulation retained in fp32. Its exponent range is much wider than fp16's and it preserves more precision and range than E4M3; fp8 E4M3 is better reserved for carefully scaled operations rather than used as the sole training format.

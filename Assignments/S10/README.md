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

The `session10.ipynb` foundation includes a 108,320-parameter character-level causal Transformer, a self-contained toy corpus, a baseline training loop, tensor-shape output, and a generated sample. In the recorded 500-step run, mean training loss fell from 3.5595 to 0.0761 (validation loss from 3.5568 to 0.0769); the repeated toy corpus makes this a pipeline check, not a generalization result.

The numerical gradient check compares `lm_head.weight[0, 0]` using a central difference with `epsilon = 0.01` on a fixed batch. The recorded backprop gradient was `-0.01331639`; the finite-difference estimate was `-0.01331568`, an absolute difference of `7.17e-7` (about `0.0054%` relative error). The small discrepancy is consistent with finite-precision arithmetic and the finite perturbation size.

For accumulation, three micro-batches of eight sequences have lengths 8, 24, and 64 (64, 192, and 512 target tokens). The unweighted average of their mean losses was `3.573558`, versus `3.600885` when weighted by token count. Their token shares should be `8.3%`, `25.0%`, and `66.7%`, rather than one third each. The corresponding full-model gradient norms were `0.883425` for the incorrect accumulation and `0.800863` for the correctly weighted gradient; their relative difference was `39.91%` of the correct gradient norm. The notebook plots the two output-head bias gradient vectors and the weights assigned by each method.

The per-step gradient-norm analysis, MFU estimate, and floating-point representation comparison remain to be completed.

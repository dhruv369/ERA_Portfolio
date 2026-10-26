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

Assignment requirements are recorded; implementation and results are not yet documented.

# Session 11: Understanding Adam and learning-rate schedules

## Assignment

Reproduce Adam by hand, then compare the hand calculations with PyTorch. Use controlled experiments to examine bias correction, warmup, learning-rate schedules, and learning-rate scaling with model width. Tune both sides of every comparison before drawing conclusions.

## Experiments and evidence

### 1. Reproduce one Adam update

Choose one weight and a sequence of five gradients. By hand, compute the first moment `m`, second moment `v`, bias-corrected moments `m_hat` and `v_hat`, and the resulting parameter step. Compare every intermediate value and the final step with PyTorch to several decimal places.

Document the Adam hyperparameters and implementation details used in both calculations, including the epsilon placement.

### 2. Measure the effect of bias correction

Disable bias correction and plot the first twenty steps with and without it. Report the step after which the difference stops mattering, and state the numerical criterion used to make that judgment.

### 3. Track update-to-weight ratios through warmup

Log the update-to-weight ratio for every layer at every step. State the ratio definition, identify when warmup stops changing it, and explain the observed behavior.

### 4. Compare cosine and WSD

Train the same model twice: once with cosine and once with WSD, using the requested 300-step setup and stopping both runs at step 200. Report both losses and state which model you would keep. Document the schedule horizon and stopping point clearly.

### 5. Sweep learning rates across widths

Sweep learning rates at widths 256, 512, and 1,024. Plot loss against learning rate and mark the minimum for each width. Based on those results, state the learning rate you would use at width 4,096 and how confident you are in that choice.

### 6. Make the comparison fair

Tune both sides before accepting a comparison. Record the tuning procedure and settings for each method so a poorly tuned baseline is not mistaken for evidence that another method is better.

## Submission

- A detailed README with the setup, methods, plots, numerical results, tuning details, and conclusions.
- Support code sufficient to reproduce the Adam checks and training experiments.
- The GitHub repository containing the README and support code.

## Status

Assignment requirements are recorded; implementation and results are not yet documented.

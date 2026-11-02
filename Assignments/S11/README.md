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

The first experiment is implemented in [`adam_by_hand.py`](./adam_by_hand.py). Run it with `python Assignments/S11/adam_by_hand.py`. It computes five Adam updates by hand, compares `m`, `v`, `m_hat`, `v_hat`, the parameter update, and the resulting weight against `torch.optim.Adam`, and asserts that every difference is below `1e-12`.

### Experiment 1: Reproduce one Adam update

The scalar starts at `0.5`; gradients are `[0.2, -0.1, 0.05, 0.3, -0.25]`. Settings are `lr=0.01`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, no weight decay, and no AMSGrad. The implementation uses the bias-corrected update

```text
m_t      = beta1 * m_(t-1) + (1 - beta1) * g_t
v_t      = beta2 * v_(t-1) + (1 - beta2) * g_t^2
m_hat_t  = m_t / (1 - beta1^t)
v_hat_t  = v_t / (1 - beta2^t)
update_t = lr * m_hat_t / (sqrt(v_hat_t) + eps)
weight_t = weight_(t-1) - update_t
```

Epsilon is added after taking the square root of the bias-corrected second moment, matching PyTorch Adam. Hand calculations and PyTorch agree below `1e-12` for all checked moments and parameter values. Representative values:

| Step | `m_t` | `v_t` | `m_hat_t` | `v_hat_t` | Update | Weight after update |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.020000000000 | 0.000040000000 | 0.200000000000 | 0.040000000000 | 0.009999999500 | 0.490000000500 |
| 2 | 0.008000000000 | 0.000049960000 | 0.042105263158 | 0.024992496248 | 0.002663370228 | 0.487336630272 |
| 3 | 0.012200000000 | 0.000052410040 | 0.045018450185 | 0.017487494999 | 0.003404291225 | 0.483932339047 |
| 4 | 0.040980000000 | 0.000142357630 | 0.119162547252 | 0.035642836110 | 0.006311805804 | 0.477620533243 |
| 5 | 0.011882000000 | 0.000204715272 | 0.029015164465 | 0.041025022502 | 0.001432519694 | 0.476188013550 |

The remaining experiments—bias-correction comparison, warmup ratios, cosine versus WSD, and learning-rate sweeps across widths—are not yet complete.

# Session 12: Simulating ZeRO across 32 virtual GPUs

## Assignment

Create a simple simulation with 32 virtual devices (for example, CPU threads or a Colab GPU) and run a demo model across them. Compare an unsharded baseline with ZeRO stages 1, 2, and 3 to show how parameter, gradient, and optimizer-state partitioning affect memory and computation.

The goal is to demonstrate your own understanding of the concepts. Explain the design and results in your own words rather than relying on an agent-written explanation.

## Experiments and evidence

### 1. Set up 32 virtual devices

Describe how the 32 virtual devices or ranks are represented and how work is assigned across them. State what is simulated and what runs on actual hardware.

### 2. Run a demo model

Use a small model and a repeatable workload that can be run under each configuration. Keep the model, workload, and relevant settings consistent so the comparisons are meaningful.

### 3. Compare the sharding stages

Compare these configurations:

- **Baseline (ZeRO-0):** Parameters, gradients, and optimizer states are replicated.
- **ZeRO-1:** Partition optimizer states across ranks.
- **ZeRO-2:** Partition optimizer states and gradients.
- **ZeRO-3:** Partition optimizer states, gradients, and parameters.

For each configuration, explain what is stored on each rank, what must be communicated, and how the work proceeds during a training step.

### 4. Show memory and computation changes

Report memory per rank and in aggregate, and explain how the memory use changes across stages. Include the assumptions and formulas behind the estimates, and distinguish measured values from theoretical estimates.

Measure or estimate computation and communication for each configuration. Explain any added work or communication caused by sharding. Do not present a virtual-device simulation as evidence of real multi-GPU speedup.

## Submission

- A Jupyter notebook named `session12.ipynb` that runs the demo and contains the comparison data, calculations, and explanations.
- A detailed README explaining the setup, the ZeRO stages, the memory and computation results, and what you learned in your own words.
- The GitHub repository containing the notebook and README.

## Status

Assignment requirements are recorded; implementation and results are not yet documented.

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

## Implementation

The runnable notebook is [`session12.ipynb`](./session12.ipynb); its support code is [`zero_simulation.py`](./zero_simulation.py). Run the notebook top-to-bottom, or run the standalone script from the repository root:

```bash
python Assignments/S12/zero_simulation.py
```

The demo is a `1 -> 64 -> 64 -> 1` tanh MLP with 4,353 FP32 parameters. A fixed synthetic regression set has 128 points, divided into 32 equal rank-local batches of four. For each of 20 updates, the simulator runs forward/backward work for every rank sequentially on CPU. It uses Adam with learning rate `0.001`, `beta1=0.9`, `beta2=0.999`, and `eps=1e-8`, applying `m_hat / (sqrt(v_hat) + eps)`. ZeRO-0 and ZeRO-1 average full rank gradients; ZeRO-2 and ZeRO-3 reduce-scatter the rank-local gradient shards. ZeRO-1 through ZeRO-3 update the parameter slices owned by each virtual rank using its Adam moment shards, then reconstruct the parameter vector; ZeRO-3 also starts each forward/backward pass from gathered parameter shards.

The four runs start from identical weights and see the same data. Each reduces MSE from `0.432807` to `0.422089`. The simulator checks that final parameter vectors agree with ZeRO-0 within `1e-7`; this verifies that the different aggregation and state-partition paths produce the same global Adam update. The virtual ranks execute sequentially in one Python process: there are no 32 worker processes, GPUs, or real network collectives. To keep the simulation small, the Python process uses temporary full-gradient and model buffers; those are not represented as per-rank persistent allocations. The model computation is a CPU simulation, while memory and communication figures below are calculated from explicit persistent storage and ring-collective models; they are **not measured device memory or speed**.

## What each stage stores and does

- **ZeRO-0 (replicated baseline):** Every rank has all parameters, gradients, and both Adam moments. Each rank computes on its local data; gradients are averaged across ranks; every replica applies the same full Adam update.
- **ZeRO-1:** Parameters and gradients remain replicated. Adam's first- and second-moment tensors are split among ranks. Gradients are averaged, each rank updates its owned parameter slice using its moment slice, and the updated parameter shards are gathered so each rank can run the next forward pass.
- **ZeRO-2:** Parameters remain replicated; gradients and Adam moments are split. Rank-local gradients are reduced into gradient shards for their owning ranks, which update their matching parameter slices. Updated parameters are gathered after the update.
- **ZeRO-3:** Parameters, gradients, and optimizer moments are all split among ranks between compute phases. Parameter shards are gathered to run the forward/backward computation; gradient shards are reduced to their owners, which update their parameter and moment shards. The next compute phase gathers parameters again.

These are the concepts modeled by the notebook. In a real implementation, parameters are typically gathered per layer and collectives are overlapped with computation; this small flat-vector simulation does not reproduce those runtime optimizations.

## Memory results

Assumptions: `P=4,353`, `N=32`, each parameter/gradient/Adam value is FP32 (`b=4` bytes), and Adam has two FP32 moment values per parameter. The table reports the busiest rank for per-rank shard memory because `P` does not divide evenly by 32 (parameter shards have 136 or 137 values). The activation estimate is a separate lower-bound-like estimate for two hidden activations and one output tensor at local batch size four; it omits activation gradients and temporary/runtime buffers.

Let `ceil_shard = ceil(P/N)`. Persistent memory on the busiest rank is:

| Stage | Parameters/rank | Gradients/rank | Adam values/rank | Persistent MiB/rank | Activation estimate MiB/rank | Total estimate MiB/rank | Aggregate persistent MiB | Aggregate total estimate MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZeRO-0 | `P` | `P` | `2P` | 0.066422 | 0.001968 | 0.068390 | 2.125488 | 2.188477 |
| ZeRO-1 | `P` | `P` | `2*ceil_shard` | 0.034256 | 0.001968 | 0.036224 | 1.095955 | 1.158943 |
| ZeRO-2 | `P` | `ceil_shard` | `2*ceil_shard` | 0.018173 | 0.001968 | 0.020142 | 0.581188 | 0.644176 |
| ZeRO-3 | `ceil_shard` | `ceil_shard` | `2*ceil_shard` | 0.002090 | 0.001968 | 0.004059 | 0.066422 | 0.129410 |

Aggregate totals use the actual full-tensor ownership counts, not `N` times the busiest rank. The persistent-state formulas across the whole job are:

```text
ZeRO-0: 4 * N * P * b       # every rank has params, grads, and 2 Adam states
ZeRO-1: (2 * N + 2) * P * b # params and grads replicated; optimizer states sharded
ZeRO-2: (N + 3) * P * b     # params replicated; grads and optimizer states sharded
ZeRO-3: 4 * P * b           # params, grads, and optimizer states each stored once
```

The corresponding rank memory falls from `0.066422 MiB` at ZeRO-0 to `0.002090 MiB` of persistent state at ZeRO-3. This toy network is so small that these savings are only kilobytes; large models make the same proportional reduction more consequential. Activations do not shrink in this simple estimate because the local batch and model computation are held constant.

## Computation and communication

For dense layers, the notebook estimates forward-plus-backward work as `6 * dense_weight_parameters * local_batch`, counting approximately two FLOPs per multiply-accumulate in forward and four in backward. This gives `101,376 FLOPs/rank/update` and `3,244,032 FLOPs across 32 ranks/update` for every stage. ZeRO partitions state but does not remove the MLP's matrix multiplications. The values are analytic estimates, not timing measurements; activations, biases, optimizer arithmetic, and kernel overhead are excluded.

Communication is estimated with a simple ring model. Per rank, an all-reduce sends `2(N-1)/N` payloads; reduce-scatter and all-gather each send `(N-1)/N` payloads. Here the FP32 parameter and gradient payload sizes are equal.

| Stage | Modeled collectives per update | Sent per rank | Aggregate sent across 32 ranks |
| --- | --- | ---: | ---: |
| ZeRO-0 | Gradient all-reduce | 32.945 KiB | 1.030 MiB |
| ZeRO-1 | Gradient all-reduce + updated-parameter all-gather | 49.418 KiB | 1.544 MiB |
| ZeRO-2 | Gradient reduce-scatter + updated-parameter all-gather | 32.945 KiB | 1.030 MiB |
| ZeRO-3 | Parameter all-gather twice + gradient reduce-scatter | 49.418 KiB | 1.544 MiB |

The extra parameter gathering can make higher ZeRO stages communication-heavy for this model. Real runtimes use layerwise gathers, overlap communication and computation, and are affected by bandwidth and latency. These estimates do not show whether any stage would be faster on actual hardware.

## What I learned

The important distinction for me is that ZeRO is a memory-partitioning strategy, not a way to make the model's arithmetic smaller. Each later stage removes a category of replicated state, so the memory required on a single rank drops, while the global training calculation still needs the full model. That requires more coordination: gradients must reach the rank that owns their optimizer state, and sharded parameters must be available when a layer computes. Whether a stage is useful depends on whether the memory it saves is worth the communication it adds. This notebook makes those trade-offs visible, but a real performance decision still needs a multi-GPU benchmark.

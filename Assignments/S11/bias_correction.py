from pathlib import Path

import matplotlib.pyplot as plt
import torch


INITIAL_WEIGHT = 1.0
CONSTANT_GRADIENT = 0.1
LEARNING_RATE = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
PLOT_STEPS = 20
MAX_STEPS = 10_000
RELATIVE_TOLERANCE = 0.01


def main():
    parameter = torch.nn.Parameter(
        torch.tensor([INITIAL_WEIGHT], dtype=torch.float64)
    )
    optimizer = torch.optim.Adam(
        [parameter],
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        eps=EPSILON,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
    )

    moment = 0.0
    squared_moment = 0.0
    corrected_weight = INITIAL_WEIGHT
    uncorrected_weight = INITIAL_WEIGHT
    corrected_updates = []
    uncorrected_updates = []
    corrected_weights = [INITIAL_WEIGHT]
    uncorrected_weights = [INITIAL_WEIGHT]
    relative_update_errors = []

    for step in range(1, MAX_STEPS + 1):
        moment = BETA1 * moment + (1 - BETA1) * CONSTANT_GRADIENT
        squared_moment = (
            BETA2 * squared_moment
            + (1 - BETA2) * CONSTANT_GRADIENT**2
        )
        corrected_moment = moment / (1 - BETA1**step)
        corrected_squared_moment = squared_moment / (1 - BETA2**step)
        corrected_update = LEARNING_RATE * corrected_moment / (
            corrected_squared_moment**0.5 + EPSILON
        )
        uncorrected_update = LEARNING_RATE * moment / (
            squared_moment**0.5 + EPSILON
        )

        corrected_weight -= corrected_update
        uncorrected_weight -= uncorrected_update
        corrected_updates.append(corrected_update)
        uncorrected_updates.append(uncorrected_update)
        corrected_weights.append(corrected_weight)
        uncorrected_weights.append(uncorrected_weight)
        relative_update_errors.append(
            abs(uncorrected_update - corrected_update) / abs(corrected_update)
        )

        parameter.grad = torch.tensor([CONSTANT_GRADIENT], dtype=torch.float64)
        previous_torch_weight = parameter.item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch_update = previous_torch_weight - parameter.item()
        if abs(torch_update - corrected_update) > 1e-12:
            raise AssertionError(
                f"PyTorch mismatch at step {step}: "
                f"torch={torch_update}, formula={corrected_update}"
            )

    threshold_step = next(
        (
            step
            for step in range(1, MAX_STEPS + 1)
            if all(
                error < RELATIVE_TOLERANCE
                for error in relative_update_errors[step - 1 :]
            )
        ),
        None,
    )
    if threshold_step is None:
        raise RuntimeError(
            f"Relative update error did not stay below {RELATIVE_TOLERANCE:.0%} "
            f"within {MAX_STEPS} steps"
        )

    print(
        f"Adam: lr={LEARNING_RATE}, beta1={BETA1}, beta2={BETA2}, "
        f"eps={EPSILON}, weight_decay=0"
    )
    print(f"Constant gradient: {CONSTANT_GRADIENT}; initial weight: {INITIAL_WEIGHT}")
    print("Both methods use identical moments and epsilon placement; only bias correction differs.")
    print("\nFirst 20 updates:")
    print("step  corrected_update  no_correction_update  relative_difference")
    for step in range(1, PLOT_STEPS + 1):
        print(
            f"{step:>4}  {corrected_updates[step - 1]:>16.10f}"
            f"  {uncorrected_updates[step - 1]:>19.10f}"
            f"  {relative_update_errors[step - 1]:>18.2%}"
        )
    print(
        f"\nCriterion: relative update-size difference "
        f"|u_no_bias - u_bias| / |u_bias| < {RELATIVE_TOLERANCE:.0%}, "
        "and it remains below the threshold thereafter."
    )
    print(
        f"First step meeting the criterion: {threshold_step} "
        f"({relative_update_errors[threshold_step - 1]:.6%} difference)."
    )
    print(
        f"At step {PLOT_STEPS}, the relative update difference is "
        f"{relative_update_errors[PLOT_STEPS - 1]:.2%}."
    )

    steps = range(1, PLOT_STEPS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        range(PLOT_STEPS + 1),
        corrected_weights[: PLOT_STEPS + 1],
        marker="o",
        label="Bias correction enabled",
    )
    axes[0].plot(
        range(PLOT_STEPS + 1),
        uncorrected_weights[: PLOT_STEPS + 1],
        marker="s",
        label="Bias correction disabled",
    )
    axes[0].set(
        title="Parameter trajectory",
        xlabel="Adam step",
        ylabel="Weight value",
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        steps,
        corrected_updates[:PLOT_STEPS],
        marker="o",
        label="Bias correction enabled",
    )
    axes[1].plot(
        steps,
        uncorrected_updates[:PLOT_STEPS],
        marker="s",
        label="Bias correction disabled",
    )
    axes[1].set(
        title="Update magnitude (first 20 steps)",
        xlabel="Adam step",
        ylabel="Update magnitude",
        xticks=list(range(1, PLOT_STEPS + 1, 2)),
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    output_path = Path(__file__).with_name("bias_correction.png")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Plot saved to {output_path.name}")


if __name__ == "__main__":
    main()

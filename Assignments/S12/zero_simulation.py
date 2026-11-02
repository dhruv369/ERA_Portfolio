from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn


WORLD_SIZE = 32
WIDTH = 64
LOCAL_BATCH_SIZE = 4
DEMO_TRAINING_STEPS = 20
DEMO_LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
FLOAT32_BYTES = torch.tensor([], dtype=torch.float32).element_size()
STAGES = ("ZeRO-0", "ZeRO-1", "ZeRO-2", "ZeRO-3")


class DemoRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.fc3 = nn.Linear(WIDTH, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def partition_counts(item_count: int, world_size: int = WORLD_SIZE) -> list[int]:
    quotient, remainder = divmod(item_count, world_size)
    return [
        quotient + (1 if rank < remainder else 0)
        for rank in range(world_size)
    ]


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    gradients = [
        parameter.grad.reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("The model produced no gradients.")
    return torch.cat(gradients)


def load_flat_parameters(model: nn.Module, flat_parameters: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(flat_parameters, model.parameters())


def adam_update(
    parameters: torch.Tensor,
    gradients: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_moment = BETA1 * first_moment + (1 - BETA1) * gradients
    second_moment = BETA2 * second_moment + (1 - BETA2) * gradients.square()
    corrected_first = first_moment / (1 - BETA1**step)
    corrected_second = second_moment / (1 - BETA2**step)
    updated_parameters = parameters - DEMO_LEARNING_RATE * corrected_first / (
        corrected_second.sqrt() + EPSILON
    )
    return updated_parameters, first_moment, second_moment


def run_demo_training() -> tuple[int, dict[str, dict[str, float]]]:
    torch.manual_seed(12)
    torch.set_num_threads(1)
    x = torch.linspace(-1, 1, WORLD_SIZE * LOCAL_BATCH_SIZE).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x) + 0.3 * x
    rank_batches = list(
        zip(
            x.split(LOCAL_BATCH_SIZE),
            y.split(LOCAL_BATCH_SIZE),
        )
    )
    results: dict[str, dict[str, float]] = {}
    final_parameter_vectors: dict[str, torch.Tensor] = {}
    parameter_count = 0

    for stage in STAGES:
        torch.manual_seed(1200)
        model = DemoRegressor()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        loss_fn = nn.MSELoss()
        initial_parameters = torch.nn.utils.parameters_to_vector(
            model.parameters()
        ).detach().clone()
        shard_sizes = partition_counts(parameter_count)
        parameter_shards = [
            shard.clone() for shard in initial_parameters.split(shard_sizes)
        ]
        parameter_vector = (
            None if stage == "ZeRO-3" else initial_parameters
        )
        first_moment_shards = [
            torch.zeros_like(shard) for shard in parameter_shards
        ]
        second_moment_shards = [
            torch.zeros_like(shard) for shard in parameter_shards
        ]
        if stage == "ZeRO-0":
            first_moment = torch.zeros_like(parameter_vector)
            second_moment = torch.zeros_like(parameter_vector)

        model.train()
        with torch.no_grad():
            initial_loss = loss_fn(model(x), y).item()

        for step in range(1, DEMO_TRAINING_STEPS + 1):
            full_parameters = (
                torch.cat(parameter_shards)
                if stage == "ZeRO-3"
                else parameter_vector
            )
            if full_parameters is None:
                raise RuntimeError(f"Missing full parameters for {stage}")
            local_gradients = []
            for rank_x, rank_y in rank_batches:
                load_flat_parameters(model, full_parameters)
                model.zero_grad(set_to_none=True)
                rank_loss = loss_fn(model(rank_x), rank_y)
                rank_loss.backward()
                local_gradients.append(flatten_gradients(model).detach().clone())

            if stage in ("ZeRO-0", "ZeRO-1"):
                reduced_gradient = torch.stack(local_gradients).mean(dim=0)
                gradient_shards = list(reduced_gradient.split(shard_sizes))
            else:
                local_gradient_shards = [
                    list(local_gradient.split(shard_sizes))
                    for local_gradient in local_gradients
                ]
                gradient_shards = [
                    torch.stack(
                        [rank_shards[rank] for rank_shards in local_gradient_shards]
                    ).mean(dim=0)
                    for rank in range(WORLD_SIZE)
                ]

            if stage == "ZeRO-0":
                if parameter_vector is None:
                    raise RuntimeError("ZeRO-0 requires a full parameter vector.")
                parameter_vector, first_moment, second_moment = adam_update(
                    parameter_vector,
                    torch.cat(gradient_shards),
                    first_moment,
                    second_moment,
                    step,
                )
            else:
                updated_shards = []
                for rank in range(WORLD_SIZE):
                    updated_shard, first_moment_shards[rank], second_moment_shards[
                        rank
                    ] = adam_update(
                        parameter_shards[rank],
                        gradient_shards[rank],
                        first_moment_shards[rank],
                        second_moment_shards[rank],
                        step,
                    )
                    updated_shards.append(updated_shard)
                parameter_shards = updated_shards
                if stage != "ZeRO-3":
                    parameter_vector = torch.cat(parameter_shards)

        final_parameters = (
            torch.cat(parameter_shards)
            if stage == "ZeRO-3"
            else parameter_vector
        )
        if final_parameters is None:
            raise RuntimeError(f"Missing final parameters for {stage}")
        with torch.no_grad():
            load_flat_parameters(model, final_parameters)
            final_loss = loss_fn(model(x), y).item()
        if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
            raise RuntimeError(f"Non-finite demo loss for {stage}")
        final_parameter_vectors[stage] = final_parameters.clone()
        results[stage] = {
            "initial_loss": initial_loss,
            "loss_after_training": final_loss,
        }

    reference_parameters = final_parameter_vectors["ZeRO-0"]
    for stage, stage_parameters in final_parameter_vectors.items():
        if not torch.allclose(stage_parameters, reference_parameters, atol=1e-7, rtol=0):
            raise AssertionError(f"{stage} produced different parameters from ZeRO-0.")
    return parameter_count, results


def build_comparison(parameter_count: int) -> pd.DataFrame:
    parameter_shards = partition_counts(parameter_count)
    max_shard_count = max(parameter_shards)

    parameter_bytes = parameter_count * FLOAT32_BYTES
    gradient_bytes = parameter_bytes
    optimizer_bytes = 2 * parameter_bytes

    per_rank_states = {
        "ZeRO-0": (parameter_count, parameter_count, 2 * parameter_count),
        "ZeRO-1": (
            parameter_count,
            parameter_count,
            2 * max_shard_count,
        ),
        "ZeRO-2": (
            parameter_count,
            max_shard_count,
            2 * max_shard_count,
        ),
        "ZeRO-3": (
            max_shard_count,
            max_shard_count,
            2 * max_shard_count,
        ),
    }

    # Simple ring estimates: AR = 2(N-1)/N, RS = AG = (N-1)/N payloads sent per rank.
    ring_factor = (WORLD_SIZE - 1) / WORLD_SIZE
    communication_bytes = {
        "ZeRO-0": 2 * ring_factor * gradient_bytes,
        "ZeRO-1": 2 * ring_factor * gradient_bytes + ring_factor * parameter_bytes,
        "ZeRO-2": ring_factor * gradient_bytes + ring_factor * parameter_bytes,
        "ZeRO-3": ring_factor * gradient_bytes + 2 * ring_factor * parameter_bytes,
    }
    communication = {
        "ZeRO-0": "gradient all-reduce",
        "ZeRO-1": "gradient all-reduce + updated-parameter all-gather",
        "ZeRO-2": "gradient reduce-scatter + updated-parameter all-gather",
        "ZeRO-3": "parameter all-gather x2 + gradient reduce-scatter",
    }

    dense_weight_count = WIDTH + WIDTH * WIDTH + WIDTH
    flops_per_rank = 6 * dense_weight_count * LOCAL_BATCH_SIZE
    activation_bytes = LOCAL_BATCH_SIZE * (2 * WIDTH + 1) * FLOAT32_BYTES
    aggregate_state_bytes = {
        "ZeRO-0": WORLD_SIZE
        * (parameter_bytes + gradient_bytes + optimizer_bytes),
        "ZeRO-1": WORLD_SIZE * (parameter_bytes + gradient_bytes)
        + optimizer_bytes,
        "ZeRO-2": WORLD_SIZE * parameter_bytes
        + gradient_bytes
        + optimizer_bytes,
        "ZeRO-3": parameter_bytes + gradient_bytes + optimizer_bytes,
    }

    records: list[dict[str, int | float | str]] = []
    for stage in STAGES:
        parameter_count_per_rank, gradient_count_per_rank, optimizer_count_per_rank = (
            per_rank_states[stage]
        )
        parameter_memory = parameter_count_per_rank * FLOAT32_BYTES
        gradient_memory = gradient_count_per_rank * FLOAT32_BYTES
        optimizer_memory = optimizer_count_per_rank * FLOAT32_BYTES
        state_memory = parameter_memory + gradient_memory + optimizer_memory

        records.append(
            {
                "stage": stage,
                "parameters_per_rank": parameter_count_per_rank,
                "gradients_per_rank": gradient_count_per_rank,
                "optimizer_values_per_rank": optimizer_count_per_rank,
                "parameter_memory_bytes_per_rank": parameter_memory,
                "gradient_memory_bytes_per_rank": gradient_memory,
                "optimizer_memory_bytes_per_rank": optimizer_memory,
                "persistent_state_bytes_per_rank": state_memory,
                "activation_estimate_bytes_per_rank": activation_bytes,
                "estimated_total_bytes_per_rank": state_memory + activation_bytes,
                "persistent_state_bytes_aggregate": aggregate_state_bytes[stage],
                "estimated_total_bytes_aggregate": aggregate_state_bytes[stage]
                + WORLD_SIZE * activation_bytes,
                "compute_flops_per_rank_per_step": flops_per_rank,
                "compute_flops_aggregate_per_step": WORLD_SIZE * flops_per_rank,
                "communication": communication[stage],
                "communication_bytes_sent_per_rank": communication_bytes[stage],
                "communication_bytes_sent_aggregate": WORLD_SIZE
                * communication_bytes[stage],
            }
        )

    frame = pd.DataFrame(records)
    if not frame["persistent_state_bytes_aggregate"].is_monotonic_decreasing:
        raise AssertionError("Aggregate persistent state should decrease by stage.")
    if frame.iloc[-1]["persistent_state_bytes_aggregate"] != (
        parameter_bytes + gradient_bytes + optimizer_bytes
    ):
        raise AssertionError("ZeRO-3 aggregate state must equal one full model state.")
    if not (
        frame["compute_flops_per_rank_per_step"] == flops_per_rank
    ).all():
        raise AssertionError("ZeRO stages must not change model arithmetic.")
    return frame


def make_plots(frame: pd.DataFrame, output_dir: Path) -> Path:
    mib = 1024**2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(
        frame["stage"],
        frame["persistent_state_bytes_per_rank"] / mib,
        label="Parameters + gradients + Adam states",
    )
    axes[0].bar(
        frame["stage"],
        frame["activation_estimate_bytes_per_rank"] / mib,
        bottom=frame["persistent_state_bytes_per_rank"] / mib,
        label="Activation estimate (additional)",
    )
    axes[0].set(
        title="Estimated memory on the busiest rank",
        ylabel="MiB per rank",
    )
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        frame["stage"],
        frame["communication_bytes_sent_per_rank"] / 1024,
        color=["tab:blue", "tab:orange", "tab:green", "tab:red"],
    )
    axes[1].set(
        title="Modeled communication per rank per step",
        ylabel="KiB sent (ring estimate)",
    )
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("32-rank ZeRO simulation (estimates, not hardware measurements)")
    fig.tight_layout()
    output_path = output_dir / "zero_memory_communication.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    output_dir = Path(__file__).parent
    parameter_count, demo_results = run_demo_training()
    frame = build_comparison(parameter_count)
    plot_path = make_plots(frame, output_dir)
    csv_path = output_dir / "zero_comparison.csv"
    frame.to_csv(csv_path, index=False)

    print(
        f"Logical world: {WORLD_SIZE} virtual ranks; no GPU processes or "
        "inter-device network are launched."
    )
    print(
        f"Demo: 1 -> {WIDTH} -> {WIDTH} -> 1 tanh MLP, "
        f"{parameter_count:,} parameters, local batch={LOCAL_BATCH_SIZE}, "
        f"{DEMO_TRAINING_STEPS} updates at LR={DEMO_LEARNING_RATE}."
    )
    for stage, losses in demo_results.items():
        print(
            f"{stage}: loss {losses['initial_loss']:.6f} -> "
            f"{losses['loss_after_training']:.6f} after CPU Adam training"
        )
    print("\nEstimated stage comparison (memory in MiB, communication in KiB/rank):")
    display_frame = frame[
        [
            "stage",
            "persistent_state_bytes_per_rank",
            "estimated_total_bytes_per_rank",
            "persistent_state_bytes_aggregate",
            "compute_flops_per_rank_per_step",
            "communication_bytes_sent_per_rank",
        ]
    ].copy()
    display_frame["persistent_state_bytes_per_rank"] /= 1024**2
    display_frame["estimated_total_bytes_per_rank"] /= 1024**2
    display_frame["persistent_state_bytes_aggregate"] /= 1024**2
    display_frame["communication_bytes_sent_per_rank"] /= 1024
    print(display_frame.to_string(index=False))
    print(f"\nCSV saved to {csv_path.name}")
    print(f"Plot saved to {plot_path.name}")


if __name__ == "__main__":
    main()

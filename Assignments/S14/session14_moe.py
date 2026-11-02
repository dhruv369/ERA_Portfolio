"""Train a linear regressor, convert it to a mixture of experts, and continue training."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

SEED = 1401
TRAIN_EXAMPLES = 2048
VALIDATION_EXAMPLES = 1024
BASELINE_STEPS = 500
MOE_STEPS = 1500
BASELINE_LEARNING_RATE = 0.05
MOE_LEARNING_RATE = 0.02
EXPERTS = 3
OUTPUT_DIR = Path(
    os.environ.get("S14_OUTPUT_DIR", Path(__file__).resolve().parent / "output")
)


class LinearRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, source: LinearRegressor) -> None:
        super().__init__()
        self.router = nn.Linear(1, EXPERTS)
        self.experts = nn.ModuleList(nn.Linear(1, 1) for _ in range(EXPERTS))

        with torch.no_grad():
            for expert in self.experts:
                expert.weight.copy_(source.linear.weight)
                expert.bias.copy_(source.linear.bias)

            # The three linear router logits favor the left, center, and right
            # input regions, respectively, with boundaries near -0.3 and 0.3.
            self.router.weight[:, 0] = torch.tensor([-8.0, 0.0, 8.0])
            self.router.bias[:] = torch.tensor([-2.4, 0.0, -2.4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights = self.router(x).softmax(dim=-1)
        expert_outputs = torch.cat([expert(x) for expert in self.experts], dim=-1)
        return (routing_weights * expert_outputs).sum(dim=-1, keepdim=True)

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x).softmax(dim=-1)

    @torch.no_grad()
    def break_expert_symmetry(self) -> None:
        symmetry_offset = (
            0.02
            * self.experts[0].weight.detach().abs().mean().clamp_min(0.1)
        )
        self.experts[0].weight.sub_(symmetry_offset)
        self.experts[2].weight.add_(symmetry_offset)
        self.experts[0].bias.add_(symmetry_offset)
        self.experts[2].bias.sub_(symmetry_offset)


def make_dataset(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(count, 1, generator=generator) * 2.0 - 1.0
    left = x[:, 0] < -0.3
    right = x[:, 0] >= 0.3
    y = torch.where(
        left,
        -2.0 * x[:, 0] - 0.8,
        torch.where(right, 2.0 * x[:, 0] - 0.4, 0.5 * x[:, 0] + 0.1),
    ).unsqueeze(1)
    return x, y


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return nn.functional.mse_loss(model(x), y).item()


def train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            losses.append(loss.item())
    return losses


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    train_x, train_y = make_dataset(TRAIN_EXAMPLES, SEED)
    validation_x, validation_y = make_dataset(VALIDATION_EXAMPLES, SEED + 1)

    linear = LinearRegressor()
    baseline_train_losses = train(
        linear,
        train_x,
        train_y,
        BASELINE_STEPS,
        BASELINE_LEARNING_RATE,
    )
    baseline_train_loss = mse(linear, train_x, train_y)
    baseline_validation_loss = mse(linear, validation_x, validation_y)

    moe = MixtureOfExperts(linear)
    with torch.no_grad():
        conversion_error = (moe(validation_x) - linear(validation_x)).abs().max().item()
    converted_validation_loss = mse(moe, validation_x, validation_y)
    if conversion_error > 1e-6:
        raise RuntimeError(f"Conversion did not preserve predictions: {conversion_error}")
    moe.break_expert_symmetry()
    symmetry_broken_validation_loss = mse(moe, validation_x, validation_y)

    post_conversion_train_losses = train(
        moe,
        train_x,
        train_y,
        MOE_STEPS,
        MOE_LEARNING_RATE,
    )
    final_train_loss = mse(moe, train_x, train_y)
    final_validation_loss = mse(moe, validation_x, validation_y)
    if not final_validation_loss < baseline_validation_loss:
        raise RuntimeError(
            "The MoE failed to reduce validation loss. "
            f"Linear={baseline_validation_loss:.6f}; MoE={final_validation_loss:.6f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "seed": SEED,
        "task": "piecewise-linear scalar regression",
        "train_examples": TRAIN_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "baseline_steps": BASELINE_STEPS,
        "moe_steps_after_conversion": MOE_STEPS,
        "baseline_learning_rate": BASELINE_LEARNING_RATE,
        "moe_learning_rate": MOE_LEARNING_RATE,
        "experts": EXPERTS,
        "linear_parameters": parameter_count(linear),
        "moe_parameters": parameter_count(moe),
        "max_abs_prediction_error_at_conversion": conversion_error,
        "validation_mse_at_conversion": converted_validation_loss,
        "validation_mse_after_symmetry_breaking": symmetry_broken_validation_loss,
        "linear_train_mse": baseline_train_loss,
        "linear_validation_mse": baseline_validation_loss,
        "moe_final_train_mse": final_train_loss,
        "moe_final_validation_mse": final_validation_loss,
        "validation_mse_reduction_percent": (
            100.0 * (baseline_validation_loss - final_validation_loss)
            / baseline_validation_loss
        ),
        "runtime": {
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "device": "CPU",
            "cpu_threads": torch.get_num_threads(),
        },
    }

    with torch.no_grad():
        grid = torch.linspace(-1, 1, 500).unsqueeze(1)
        left = grid[:, 0] < -0.3
        right = grid[:, 0] >= 0.3
        target_grid = torch.where(
            left,
            -2.0 * grid[:, 0] - 0.8,
            torch.where(right, 2.0 * grid[:, 0] - 0.4, 0.5 * grid[:, 0] + 0.1),
        ).unsqueeze(1)
        linear_predictions = linear(grid)
        moe_predictions = moe(grid)
        gate = moe.routing_weights(grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(train_x[:, 0], train_y[:, 0], s=6, alpha=0.15, label="train data")
    axes[0].plot(grid[:, 0], target_grid[:, 0], color="black", linewidth=2, label="target")
    axes[0].plot(grid[:, 0], linear_predictions[:, 0], label="trained linear")
    axes[0].plot(grid[:, 0], moe_predictions[:, 0], label="continued-trained MoE")
    axes[0].set(title="The MoE fits the piecewise-linear target", xlabel="x", ylabel="y")
    axes[0].legend()

    axes[1].plot(grid[:, 0], gate[:, 0], label="left expert")
    axes[1].plot(grid[:, 0], gate[:, 1], label="center expert")
    axes[1].plot(grid[:, 0], gate[:, 2], label="right expert")
    axes[1].set(title="Learned router probabilities", xlabel="x", ylabel="probability")
    axes[1].legend()
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "linear_to_moe.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    loss_history_path = OUTPUT_DIR / "session14_loss_history.csv"
    with loss_history_path.open("w", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["phase", "step", "training_mse"])
        writer.writerows(
            ("linear", min(index * 10, BASELINE_STEPS), loss)
            for index, loss in enumerate(baseline_train_losses)
        )
        writer.writerows(
            ("moe_after_conversion", min(index * 10, MOE_STEPS), loss)
            for index, loss in enumerate(post_conversion_train_losses)
        )

    loss_figure, loss_axis = plt.subplots(figsize=(7, 4.5))
    loss_axis.plot(
        np.arange(len(baseline_train_losses)) * 10,
        baseline_train_losses,
        label="linear baseline",
    )
    loss_axis.plot(
        BASELINE_STEPS + np.arange(len(post_conversion_train_losses)) * 10,
        post_conversion_train_losses,
        label="MoE after conversion",
    )
    loss_axis.set(
        title="Training loss continues to fall after conversion",
        xlabel="optimization steps (MoE phase starts at step 500)",
        ylabel="training MSE",
        yscale="log",
    )
    loss_axis.legend()
    loss_figure.tight_layout()
    loss_figure_path = OUTPUT_DIR / "session14_training_loss.png"
    loss_figure.savefig(loss_figure_path, dpi=160)
    plt.close(loss_figure)

    result["figure"] = str(figure_path.relative_to(OUTPUT_DIR.parent))
    result["loss_figure"] = str(loss_figure_path.relative_to(OUTPUT_DIR.parent))
    result["loss_history"] = str(loss_history_path.relative_to(OUTPUT_DIR.parent))
    result_path = OUTPUT_DIR / "session14_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Plot: {figure_path}")
    return result


if __name__ == "__main__":
    main()

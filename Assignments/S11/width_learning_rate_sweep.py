import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


WIDTHS = (256, 512, 1024)
LEARNING_RATES = (
    0.0001,
    0.001,
    0.003,
    0.005,
    0.006,
    0.007,
    0.0075,
    0.008,
    0.009,
    0.01,
    0.012,
    0.015,
    0.018,
    0.02,
    0.022,
    0.025,
    0.03,
    0.1,
)
SEEDS = (11, 23, 37, 41, 53)
TRAINING_STEPS = 200
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


class WidthRegressor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def make_dataset():
    x = torch.linspace(-1, 1, 320).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x) + 0.3 * x
    validation_mask = torch.arange(len(x)) % 5 == 0
    return x[~validation_mask], y[~validation_mask], x[validation_mask], y[validation_mask]


def run_trial(width, learning_rate, seed, train_x, train_y, validation_x, validation_y):
    torch.manual_seed(seed)
    model = WidthRegressor(width)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(BETA1, BETA2),
        eps=EPSILON,
        weight_decay=0,
    )
    loss_fn = nn.MSELoss()

    for step in range(1, TRAINING_STEPS + 1):
        model.train()
        train_loss = loss_fn(model(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        if not torch.isfinite(train_loss):
            raise RuntimeError(
                f"Non-finite training loss at width={width}, "
                f"learning_rate={learning_rate}, seed={seed}, step={step}"
            )

    model.eval()
    with torch.no_grad():
        final_train_loss = loss_fn(model(train_x), train_y).item()
        validation_loss = loss_fn(model(validation_x), validation_y).item()
    if not math.isfinite(final_train_loss) or not math.isfinite(validation_loss):
        raise RuntimeError(
            f"Non-finite final loss at width={width}, "
            f"learning_rate={learning_rate}, seed={seed}"
        )
    return final_train_loss, validation_loss


def main():
    torch.set_num_threads(1)
    train_x, train_y, validation_x, validation_y = make_dataset()
    records = []

    for width in WIDTHS:
        for learning_rate in LEARNING_RATES:
            for seed in SEEDS:
                train_loss, validation_loss = run_trial(
                    width,
                    learning_rate,
                    seed,
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                )
                records.append(
                    {
                        "width": width,
                        "learning_rate": learning_rate,
                        "seed": seed,
                        "training_steps": TRAINING_STEPS,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                    }
                )

    output_dir = Path(__file__).parent
    csv_path = output_dir / "width_learning_rate_sweep.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    summaries = {}
    for width in WIDTHS:
        summaries[width] = {}
        for learning_rate in LEARNING_RATES:
            matching = [
                row
                for row in records
                if row["width"] == width and row["learning_rate"] == learning_rate
            ]
            losses = [row["validation_loss"] for row in matching]
            mean_loss = sum(losses) / len(losses)
            variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
            summaries[width][learning_rate] = (mean_loss, math.sqrt(variance))

    minima = {
        width: min(
            summaries[width],
            key=lambda learning_rate: summaries[width][learning_rate][0],
        )
        for width in WIDTHS
    }
    annotation_offsets = {256: (20, -20), 512: (30, 25), 1024: (-110, 0)}

    print(
        f"Model: 1 -> width -> width -> 1 tanh MLP; "
        f"train={len(train_x)}, validation={len(validation_x)} fixed examples"
    )
    print(
        f"Adam: beta1={BETA1}, beta2={BETA2}, eps={EPSILON}, weight_decay=0; "
        f"training steps={TRAINING_STEPS}; seeds={SEEDS}"
    )
    print(f"Learning-rate candidates: {LEARNING_RATES}")
    print("Mean validation MSE +/- population SD across seeds:")
    print("width  " + "  ".join(f"lr={lr:g}" for lr in LEARNING_RATES))
    for width in WIDTHS:
        values = "  ".join(
            f"{summaries[width][lr][0]:.6g}+/-{summaries[width][lr][1]:.2g}"
            for lr in LEARNING_RATES
        )
        print(f"{width:>5}  {values}")
    print("Minimum mean validation MSE by width:")
    for width in WIDTHS:
        lr = minima[width]
        mean_loss, std_loss = summaries[width][lr]
        print(
            f"width={width}: lr={lr:g}, validation_mse={mean_loss:.8g} "
            f"+/- {std_loss:.2g}"
        )
    print(f"Per-seed results saved to {csv_path.name} ({len(records)} trials).")

    fig, axis = plt.subplots(figsize=(8, 5))
    for width in WIDTHS:
        means = [summaries[width][lr][0] for lr in LEARNING_RATES]
        standard_deviations = [summaries[width][lr][1] for lr in LEARNING_RATES]
        axis.errorbar(
            LEARNING_RATES,
            means,
            yerr=standard_deviations,
            marker="o",
            capsize=3,
            label=f"width {width}",
        )
        minimum_lr = minima[width]
        minimum_loss = summaries[width][minimum_lr][0]
        axis.scatter(
            [minimum_lr],
            [minimum_loss],
            marker="*",
            s=180,
            edgecolors="black",
            zorder=5,
        )
        axis.annotate(
            f"min LR={minimum_lr:g}",
            (minimum_lr, minimum_loss),
            xytext=annotation_offsets[width],
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )

    axis.set(
        title=f"Validation MSE after {TRAINING_STEPS} Adam updates",
        xlabel="Learning rate",
        ylabel="Validation MSE (mean +/- SD)",
        xscale="log",
        yscale="log",
    )
    axis.grid(alpha=0.3, which="both")
    axis.legend()
    fig.tight_layout()
    plot_path = output_dir / "width_learning_rate_sweep.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Plot saved to {plot_path.name}")


if __name__ == "__main__":
    main()

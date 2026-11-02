import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


SEED = 11
TOTAL_STEPS = 300
STOP_STEP = 200
WARMUP_STEPS = 30
WSD_DECAY_START = 240
LEARNING_RATES = (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


class SmallRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def learning_rate_at_step(schedule, peak_learning_rate, step):
    if step <= WARMUP_STEPS:
        return peak_learning_rate * step / WARMUP_STEPS
    if schedule == "cosine":
        progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        return peak_learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    if schedule == "wsd":
        if step <= WSD_DECAY_START:
            return peak_learning_rate
        decay_progress = (step - WSD_DECAY_START) / (
            TOTAL_STEPS - WSD_DECAY_START
        )
        return peak_learning_rate * (1 - decay_progress)
    raise ValueError(f"Unknown schedule: {schedule}")


def make_dataset():
    x = torch.linspace(-1, 1, 320).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x) + 0.3 * x
    validation_mask = torch.arange(len(x)) % 5 == 0
    return x[~validation_mask], y[~validation_mask], x[validation_mask], y[validation_mask]


def run_trial(schedule, peak_learning_rate, train_x, train_y, validation_x, validation_y):
    torch.manual_seed(SEED)
    model = SmallRegressor()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=peak_learning_rate / WARMUP_STEPS,
        betas=(BETA1, BETA2),
        eps=EPSILON,
        weight_decay=0,
    )
    train_loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        initial_train_loss = train_loss_fn(model(train_x), train_y).item()
        initial_validation_loss = train_loss_fn(
            model(validation_x), validation_y
        ).item()
    history = [
        {
            "schedule": schedule,
            "peak_learning_rate": peak_learning_rate,
            "step": 0,
            "learning_rate": 0.0,
            "train_loss": initial_train_loss,
            "validation_loss": initial_validation_loss,
        }
    ]

    for step in range(1, STOP_STEP + 1):
        learning_rate = learning_rate_at_step(schedule, peak_learning_rate, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        model.train()
        train_loss = train_loss_fn(model(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            measured_train_loss = train_loss_fn(model(train_x), train_y).item()
            validation_loss = train_loss_fn(model(validation_x), validation_y).item()
        if not math.isfinite(measured_train_loss) or not math.isfinite(validation_loss):
            raise RuntimeError(
                f"Non-finite loss for {schedule} at learning rate "
                f"{peak_learning_rate} on step {step}"
            )
        history.append(
            {
                "schedule": schedule,
                "peak_learning_rate": peak_learning_rate,
                "step": step,
                "learning_rate": learning_rate,
                "train_loss": measured_train_loss,
                "validation_loss": validation_loss,
            }
        )
    return history


def main():
    torch.set_num_threads(1)
    train_x, train_y, validation_x, validation_y = make_dataset()
    all_histories = []
    selected = {}

    for schedule in ("cosine", "wsd"):
        trials = []
        for peak_learning_rate in LEARNING_RATES:
            history = run_trial(
                schedule,
                peak_learning_rate,
                train_x,
                train_y,
                validation_x,
                validation_y,
            )
            all_histories.extend(history)
            trials.append(history)
        selected[schedule] = min(
            trials, key=lambda history: history[-1]["validation_loss"]
        )

    output_dir = Path(__file__).parent
    csv_path = output_dir / "cosine_vs_wsd.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_histories[0].keys())
        writer.writeheader()
        writer.writerows(all_histories)

    print(
        f"Model: 1 -> 32 -> 32 -> 1 tanh MLP; "
        f"train={len(train_x)}, validation={len(validation_x)} fixed examples"
    )
    print(
        f"Adam: beta1={BETA1}, beta2={BETA2}, eps={EPSILON}, weight_decay=0; "
        f"seed={SEED}"
    )
    print(
        f"Schedule horizon={TOTAL_STEPS}; both schedules stop after update "
        f"{STOP_STEP}; linear warmup={WARMUP_STEPS} updates; "
        f"WSD decay begins at update {WSD_DECAY_START}."
    )
    print(f"Learning-rate tuning candidates: {LEARNING_RATES}")
    print("Selected runs, ranked by validation MSE at the stopping step:")
    print("schedule  peak_lr  train_loss  validation_loss")
    for schedule, history in selected.items():
        final = history[-1]
        print(
            f"{schedule:>7}  {final['peak_learning_rate']:.4f}  "
            f"{final['train_loss']:.8f}  {final['validation_loss']:.8f}"
        )
    print(f"All tuning curves saved to {csv_path.name} ({len(all_histories)} rows).")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"cosine": "tab:blue", "wsd": "tab:orange"}
    for schedule, history in selected.items():
        steps = [row["step"] for row in history]
        train_losses = [row["train_loss"] for row in history]
        validation_losses = [row["validation_loss"] for row in history]
        axes[0].plot(
            steps,
            train_losses,
            color=colors[schedule],
            label=f"{schedule} train",
        )
        axes[0].plot(
            steps,
            validation_losses,
            color=colors[schedule],
            linestyle="--",
            label=f"{schedule} validation",
        )
        axes[1].plot(
            [row["step"] for row in history[1:]],
            [row["learning_rate"] for row in history[1:]],
            color=colors[schedule],
            label=f"{schedule} (peak LR {history[-1]['peak_learning_rate']})",
        )

    axes[0].set(
        title="Best-tuned schedule losses through update 200",
        xlabel="Optimizer update",
        ylabel="MSE",
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].axvline(
        WSD_DECAY_START,
        color="black",
        linestyle=":",
        label="WSD decay starts (240)",
    )
    axes[1].set(
        title="Learning rates (300-update horizon)",
        xlabel="Optimizer update",
        ylabel="Learning rate",
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    plot_path = output_dir / "cosine_vs_wsd.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Plot saved to {plot_path.name}")


if __name__ == "__main__":
    main()

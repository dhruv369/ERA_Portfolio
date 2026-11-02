import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


SEED = 11
BASE_LEARNING_RATE = 0.01
WARMUP_STEPS = 50
TOTAL_STEPS = 150
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


def layer_parameter_groups(model):
    groups = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            groups[name] = tuple(
                parameter
                for parameter in (module.weight, module.bias)
                if parameter is not None
            )
    return groups


def layer_norm(parameters):
    return torch.sqrt(sum(parameter.detach().square().sum() for parameter in parameters))


def main():
    torch.manual_seed(SEED)
    model = SmallRegressor()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=BASE_LEARNING_RATE / WARMUP_STEPS,
        betas=(BETA1, BETA2),
        eps=EPSILON,
        weight_decay=0,
    )
    groups = layer_parameter_groups(model)

    x = torch.linspace(-1, 1, 256).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x) + 0.3 * x
    records = []
    losses = []

    for step in range(1, TOTAL_STEPS + 1):
        learning_rate = BASE_LEARNING_RATE * min(step / WARMUP_STEPS, 1.0)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        before = {
            name: tuple(parameter.detach().clone() for parameter in parameters)
            for name, parameters in groups.items()
        }

        prediction = model(x)
        loss = nn.functional.mse_loss(prediction, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        for name, parameters in groups.items():
            old_parameters = before[name]
            update_norm = torch.sqrt(
                sum(
                    (parameter.detach() - old_parameter).square().sum()
                    for parameter, old_parameter in zip(parameters, old_parameters)
                )
            )
            parameter_norm = layer_norm(old_parameters)
            update_weight_ratio = update_norm / parameter_norm.clamp_min(1e-12)
            records.append(
                {
                    "step": step,
                    "layer": name,
                    "learning_rate": learning_rate,
                    "update_norm": update_norm.item(),
                    "weight_norm_before_update": parameter_norm.item(),
                    "update_to_weight_ratio": update_weight_ratio.item(),
                }
            )

    output_dir = Path(__file__).parent
    csv_path = output_dir / "update_weight_ratio.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(
        f"Model: 1 -> 32 -> 32 -> 1 tanh MLP; fixed full batch: {len(x)} examples"
    )
    print(
        f"Adam: beta1={BETA1}, beta2={BETA2}, eps={EPSILON}, weight_decay=0"
    )
    print(
        f"Linear warmup: {BASE_LEARNING_RATE / WARMUP_STEPS:.6f} -> "
        f"{BASE_LEARNING_RATE:.6f} over updates 1-{WARMUP_STEPS}; "
        f"constant thereafter through update {TOTAL_STEPS}"
    )
    print(f"Initial loss: {losses[0]:.6f}; final loss: {losses[-1]:.6f}")
    print(
        "ratio = ||parameters_after - parameters_before||_2 / "
        "||parameters_before||_2; weight and bias are grouped per Linear layer."
    )
    print("Selected measurements:")
    print("step  learning_rate  layer  update_to_weight_ratio")
    for step in (1, 10, 25, 49, 50, 51, 75, 100, 150):
        for name in groups:
            record = next(
                row for row in records if row["step"] == step and row["layer"] == name
            )
            print(
                f"{step:>4}  {record['learning_rate']:.6f}      "
                f"{name:>4}  {record['update_to_weight_ratio']:.8f}"
            )
    print(
        f"Learning rate reaches its plateau on update {WARMUP_STEPS}; "
        f"update {WARMUP_STEPS + 1} is the first update with no warmup scaling."
    )
    print(
        "The measured ratios can still vary after that point because Adam's "
        "moment estimates and the layer weight norms continue changing."
    )
    print(f"Per-layer, per-step log saved to {csv_path.name} ({len(records)} rows).")

    steps = range(1, TOTAL_STEPS + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name in groups:
        layer_records = [row for row in records if row["layer"] == name]
        ratios = [row["update_to_weight_ratio"] for row in layer_records]
        axes[0].plot(steps, ratios, label=name)
    axes[0].axvline(WARMUP_STEPS, color="black", linestyle="--", label="warmup ends")
    axes[0].set_ylabel("Update / weight norm")
    axes[0].set_title("Layerwise update-to-weight ratios")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    learning_rates = [
        BASE_LEARNING_RATE * min(step / WARMUP_STEPS, 1.0)
        for step in steps
    ]
    axes[1].plot(steps, learning_rates, color="black")
    axes[1].axvline(WARMUP_STEPS, color="black", linestyle="--")
    axes[1].set(xlabel="Optimizer update", ylabel="Learning rate")
    axes[1].set_title("Linear warmup followed by constant learning rate")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "update_weight_ratio.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Plot saved to {plot_path.name}")


if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def generate_flower_data(num_points=300, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2 * np.pi, size=num_points)
    r = 0.8 + 0.25 * np.sin(5 * t) + noise * rng.normal(size=num_points)
    x = r * np.cos(t)
    y = r * np.sin(t)
    print(f"Generated {num_points} points for the flower shape with noise level {noise}.")
    print(f"Sample of generated points (first 5):\n{np.column_stack([x, y])[:5]}")
    
    return t.astype(np.float32), np.column_stack([x, y]).astype(np.float32)


def train_test_split(t, xy, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(t))
    rng.shuffle(indices)

    split = int(len(t) * (1 - test_fraction))
    train_idx = indices[:split]
    test_idx = indices[split:]

    return t[train_idx], xy[train_idx], t[test_idx], xy[test_idx]


class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, x):
        return self.linear(x)


class ReLURegressor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_model(model, train_t, train_xy, epochs=4000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_t = torch.tensor(train_t, dtype=torch.float32).view(-1, 1)
    train_xy = torch.tensor(train_xy, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(train_t)
        loss = criterion(pred, train_xy)
        loss.backward()
        optimizer.step()

    return model


def evaluate_model(model, test_t, test_xy):
    with torch.no_grad():
        test_t = torch.tensor(test_t, dtype=torch.float32).view(-1, 1)
        test_xy = torch.tensor(test_xy, dtype=torch.float32)
        pred = model(test_t)
        mse = nn.functional.mse_loss(pred, test_xy).item()
        return pred.numpy(), mse


def plot_results(true_xy, linear_pred, relu_pred, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(true_xy[:, 0], true_xy[:, 1], s=12, c='tab:green', alpha=0.8)
    axes[0].set_title('True flower shape')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(-1.4, 1.4)
    axes[0].set_ylim(-1.4, 1.4)

    axes[1].scatter(linear_pred[:, 0], linear_pred[:, 1], s=12, c='tab:orange', alpha=0.8)
    axes[1].set_title('Linear model prediction')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(-1.4, 1.4)
    axes[1].set_ylim(-1.4, 1.4)

    axes[2].scatter(relu_pred[:, 0], relu_pred[:, 1], s=12, c='tab:blue', alpha=0.8)
    axes[2].set_title('ReLU model prediction')
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].set_xlim(-1.4, 1.4)
    axes[2].set_ylim(-1.4, 1.4)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig('flower_regression_comparison.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    t, xy = generate_flower_data(num_points=300, noise=0.05)
    train_t, train_xy, test_t, test_xy = train_test_split(t, xy, test_fraction=0.2)

    linear_model = train_model(LinearRegressor(), train_t, train_xy, epochs=4000, lr=1e-3)
    relu_model = train_model(ReLURegressor(hidden_dim=64), train_t, train_xy, epochs=4000, lr=1e-3)

    linear_pred, linear_mse = evaluate_model(linear_model, test_t, test_xy)
    relu_pred, relu_mse = evaluate_model(relu_model, test_t, test_xy)

    print(f'Linear test MSE: {linear_mse:.4f}')
    print(f'ReLU test MSE: {relu_mse:.4f}')

    plot_results(
        test_xy,
        linear_pred,
        relu_pred,
        'Flower shape regression: linear vs ReLU'
    )


import torch


INITIAL_WEIGHT = 0.5
GRADIENTS = (0.2, -0.1, 0.05, 0.3, -0.25)
LEARNING_RATE = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


def main():
    weight = INITIAL_WEIGHT
    first_moment = 0.0
    second_moment = 0.0
    parameter = torch.nn.Parameter(torch.tensor([INITIAL_WEIGHT], dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [parameter],
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        eps=EPSILON,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
    )
    previous_torch_weight = parameter.item()

    print(
        f"Adam: lr={LEARNING_RATE}, beta1={BETA1}, beta2={BETA2}, "
        f"eps={EPSILON}, weight_decay=0, amsgrad=False"
    )
    print("Equations: m=beta1*m+(1-beta1)*g; v=beta2*v+(1-beta2)*g^2")
    print("           update=lr*m_hat/(sqrt(v_hat)+eps); weight -= update")
    print("PyTorch epsilon placement: sqrt(v_hat) + eps")
    print(f"Initial weight: {INITIAL_WEIGHT:.12f}")

    for step, gradient in enumerate(GRADIENTS, start=1):
        first_moment = BETA1 * first_moment + (1 - BETA1) * gradient
        second_moment = BETA2 * second_moment + (1 - BETA2) * gradient**2
        first_moment_hat = first_moment / (1 - BETA1**step)
        second_moment_hat = second_moment / (1 - BETA2**step)
        update = LEARNING_RATE * first_moment_hat / (
            second_moment_hat**0.5 + EPSILON
        )
        weight -= update

        parameter.grad = torch.tensor([gradient], dtype=torch.float64)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        state = optimizer.state[parameter]
        torch_first_moment = state["exp_avg"].item()
        torch_second_moment = state["exp_avg_sq"].item()
        torch_first_moment_hat = torch_first_moment / (1 - BETA1**step)
        torch_second_moment_hat = torch_second_moment / (1 - BETA2**step)
        torch_update = previous_torch_weight - parameter.item()
        previous_torch_weight = parameter.item()

        manual_values = (
            first_moment,
            second_moment,
            first_moment_hat,
            second_moment_hat,
            update,
            weight,
        )
        torch_values = (
            torch_first_moment,
            torch_second_moment,
            torch_first_moment_hat,
            torch_second_moment_hat,
            torch_update,
            parameter.item(),
        )
        labels = ("m", "v", "m_hat", "v_hat", "update", "weight")

        print(f"\nstep {step}; gradient={gradient:+.6f}")
        for label, manual, reference in zip(labels, manual_values, torch_values):
            difference = abs(manual - reference)
            print(
                f"  {label:7s} hand={manual:+.12f} "
                f"PyTorch={reference:+.12f} abs_diff={difference:.3e}"
            )
            if difference > 1e-12:
                raise AssertionError(
                    f"{label} mismatch at step {step}: "
                    f"hand={manual}, PyTorch={reference}"
                )


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F


def soft_sin(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + abs(x))


ACTIVATION_FUNCTION_MODE = [
    ACTIVATION_FUNCTION_RELU,
    ACTIVATION_FUNCTION_SOFT_SIN,
] = range(2)


ACTIVATION_FUNCTION_SETTING = {
    ACTIVATION_FUNCTION_RELU: F.relu,
    ACTIVATION_FUNCTION_SOFT_SIN: soft_sin,
}

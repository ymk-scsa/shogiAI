import torch
import torch.nn.functional as F


def soft_sin(x: torch.Tensor) -> torch.Tensor:
    return 2 * x / (1 + abs(x))


def scaled_arc_tahn(x: torch.Tensor) -> torch.Tensor:
    return 2 * x / (1 + abs(x))


def quafratic(x: torch.Tensor) -> torch.Tensor:
    return x * (2 - x)


ACTIVATION_FUNCTION_MODE = [
    ACTIVATION_FUNCTION_RELU,
    ACTIVATION_FUNCTION_SOFT_SIN,
    ACTIVATION_FUNCTION_SCALED_ARC_TAHN,
    ACTIVATION_FUNCTION_QUADRATIC,
] = range(4)


ACTIVATION_FUNCTION_SETTING = {
    ACTIVATION_FUNCTION_RELU: F.relu,
    ACTIVATION_FUNCTION_SOFT_SIN: soft_sin,
    ACTIVATION_FUNCTION_SCALED_ARC_TAHN: scaled_arc_tahn,
    ACTIVATION_FUNCTION_QUADRATIC: quafratic,
}

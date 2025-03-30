import torch
import torch.nn as nn

from app.domain.features import MOVE_PLANES_NUM, MOVE_LABELS_NUM
from app.domain.activation_function import ACTIVATION_FUNCTION_SETTING


class Bias(nn.Module):
    def __init__(self, shape: int) -> None:
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.bias


# ニューラルネットワーク構築class
class ResNetBlock(nn.Module):
    def __init__(self, channels: int, activation_function_mode: int = 0):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.activation_function = ACTIVATION_FUNCTION_SETTING[activation_function_mode]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_function(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return self.activation_function(out + x)


class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        input_features: int,
        activation_function_mode: int = 0,
        blocks: int = 10,
        channels: int = 192,
        fcl: int = 256,
    ):
        super(PolicyValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_features, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)

        # resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels, activation_function_mode) for _ in range(blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(MOVE_LABELS_NUM)

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MOVE_PLANES_NUM)
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

        # activation function
        self.activation_function = ACTIVATION_FUNCTION_SETTING[activation_function_mode]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.activation_function(self.norm1(x))

        # resnet blocks
        x = self.blocks(x)

        # policy head
        policy = self.policy_conv(x)
        policy = self.policy_bias(torch.flatten(policy, 1))

        # value head
        value = self.activation_function(self.value_norm1(self.value_conv1(x)))
        value = self.activation_function(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)

        return policy, value
        # dlshogi 1/6

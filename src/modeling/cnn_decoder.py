from __future__ import annotations

import torch
from torch import nn


class CNNDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        output_size: int,
        downsample_stride: int,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.downsample_stride = int(downsample_stride)
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.activation = nn.GELU()
        self.pool = nn.AvgPool1d(
            kernel_size=self.downsample_stride,
            stride=self.downsample_stride,
        )
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden states with shape (batch, seq_len, hidden), got {hidden_states.shape}"
            )

        x = hidden_states.transpose(1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        if x.shape[-1] % self.downsample_stride != 0:
            raise ValueError(
                "Token sequence length must be divisible by cnn_downsample_stride. "
                f"Got seq_len={x.shape[-1]}, stride={self.downsample_stride}."
            )

        x = self.pool(x)
        x = x.transpose(1, 2)
        return self.classifier(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Copied constants from reference
d = 128
h = 1024
u = 32

class SC(nn.Module):
    """
    Separable Convolution Block from deep-audio-fingerprinting.
    """
    def __init__(
        self,
        input_shape,
        in_channels,
        out_channels,
        kernel_sizes,
        padding_sizes,
        stride_sizes,
        norm="layer_norm2d"
    ):
        super(SC, self).__init__()

        C, H, W = input_shape
        out_H = int(np.floor((H + 2 * padding_sizes[1][0] - kernel_sizes[1][0]) / stride_sizes[1][0] + 1))
        out_W = int(np.floor((W + 2 * padding_sizes[0][1] - kernel_sizes[0][1]) / stride_sizes[0][1] + 1))

        self.separable_conv2d_1x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[0],
            stride=stride_sizes[0],
            padding=padding_sizes[0]
        )
        self.separable_conv2d_3x1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[1],
            stride=stride_sizes[1],
            padding=padding_sizes[1]
        )

        if norm == 'layer_norm2d':
            self.BN_1x3 = nn.LayerNorm((out_channels, H, out_W))
            self.BN_3x1 = nn.LayerNorm((out_channels, out_H, out_W))
        elif norm == 'batch_norm':
            self.BN_1x3 = nn.BatchNorm2d(out_channels)
            self.BN_3x1 = nn.BatchNorm2d(out_channels)

        self.separable_conv2d = nn.Sequential(
            self.separable_conv2d_1x3, nn.ReLU(), self.BN_1x3, self.separable_conv2d_3x1, nn.ReLU(), self.BN_3x1
        )

    def forward(self, x):
        return self.separable_conv2d(x)


class SpeakerEncoder(nn.Module):
    """
    Speaker Encoder based on the reference Encoder architecture.
    Outputs a fixed-size embedding vector.
    """
    def __init__(
        self,
        embedding_dim=256,
        shapes=[
            (1, 256, 32), (d, 128, 16), (d, 64, 8), (2 * d, 32, 4), (2 * d, 16, 2), (4 * d, 8, 1), (4 * d, 4, 1),
            (h, 2, 1)
        ],
        channel_seq=[1, d, d, 2 * d, 2 * d, 4 * d, 4 * d, h, h],
        kernel_seq=[[(1, 3), (3, 1)] for i in range(8)],
        stride_seq=[[(1, 2), (2, 1)] for i in range(8)],
        pad_seq=[[(0, 1), (1, 0)] for i in range(8)]
    ):
        super(SpeakerEncoder, self).__init__()

        self.encoder = nn.Sequential()
        for i in range(len(channel_seq) - 1):
            self.encoder.append(
                SC(
                    input_shape=shapes[i],
                    in_channels=channel_seq[i],
                    out_channels=channel_seq[i + 1],
                    kernel_sizes=kernel_seq[i],
                    padding_sizes=pad_seq[i],
                    stride_sizes=stride_seq[i]
                )
            )
        
        # Output of encoder is flattened (B, 1024 * 1 = 1024)
        self.fc = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        # x shape: (B, 1, 256, 32)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # L2 Normalize embeddings
        return F.normalize(x, p=2, dim=1)

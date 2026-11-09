
import torch
import torch.nn as nn


class DynamicCNN(nn.Module):
    def __init__(
        self,
        num_layers=2,
        filters=32,
        kernel_size=3,
        dropout=0.2,
        num_classes=10
    ):
        super().__init__()

        if not 1 <= num_layers <= 5:
            raise ValueError("num_layers must be between 1 and 5")

        if filters <= 0:
            raise ValueError("filters must be positive")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        layers = []
        in_channels = 3

        for i in range(num_layers):
            out_channels = filters * (2 ** i)

            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

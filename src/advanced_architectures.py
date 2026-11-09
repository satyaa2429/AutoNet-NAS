
import torch
import torch.nn as nn


class AdvancedCNN(nn.Module):
    """
    CNN for CIFAR-10 classification.

    Improvements:
    - Multiple convolutional blocks
    - Batch normalization
    - ReLU activation
    - Max pooling
    - Dropout
    - Adaptive average pooling
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32 x 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 16 x 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            # Block 3: 8 x 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = AdvancedCNN()

    sample = torch.randn(2, 3, 32, 32)
    model.eval()

    with torch.inference_mode():
        output = model(sample)

    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Advanced CNN created successfully")
    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:", f"{parameters:,}")

    assert output.shape == (2, 10)

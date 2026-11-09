
import torch
from src.architectures import DynamicCNN

configurations = [
    {"num_layers": 2, "filters": 16, "kernel_size": 3, "dropout": 0.2},
    {"num_layers": 3, "filters": 32, "kernel_size": 3, "dropout": 0.3},
    {"num_layers": 4, "filters": 32, "kernel_size": 5, "dropout": 0.4}
]

dummy_images = torch.randn(4, 3, 32, 32)

for index, config in enumerate(configurations, start=1):
    model = DynamicCNN(**config)
    model.eval()

    with torch.no_grad():
        output = model(dummy_images)

    parameters = sum(p.numel() for p in model.parameters())

    print(f"\nCNN Architecture {index}")
    print(f"Configuration: {config}")
    print(f"Trainable parameters: {parameters:,}")
    print(f"Output shape: {tuple(output.shape)}")

    assert output.shape == (4, 10)

print("\nAll CNN architecture tests passed!")


import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = "./data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


def load_data(batch_size=32):
    print("Loading CIFAR-10 dataset...")

    full_train = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    # Reproducible dataset split
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        len(full_train),
        generator=generator
    ).tolist()

    train_dataset = Subset(full_train, indices[:4000])
    val_dataset = Subset(full_train, indices[4000:5000])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

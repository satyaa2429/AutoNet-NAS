
print("Starting CIFAR-10 dataset test...")

from src.dataset import load_data

train_loader, val_loader, test_loader = load_data()

images, labels = next(iter(train_loader))

print("\nFirst training batch:")
print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("First 10 labels:", labels[:10].tolist())

assert images.shape[1:] == (3, 32, 32)

print("\nCIFAR-10 dataset test passed!")

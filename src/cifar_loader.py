import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def load_cifar10_data(batch_size=32, data_dir='./data'):
    """
    Load CIFAR-10 dataset with basic transformations

    Args:
        batch_size (int): Batch size for data loaders
        data_dir (str): Directory to store/load data

    Returns:
        tuple: (train_loader, test_loader, classes)
    """

    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


def visualize_samples(data_loader, classes, num_samples=8):
    """
    Visualize random samples from the dataset

    Args:
        data_loader: PyTorch DataLoader
        classes: List of class names
        num_samples: Number of samples to display
    """

    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Denormalize images for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.ravel()

    for i in range(num_samples):
        # Denormalize
        img = images[i] * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)

        # Convert to numpy and transpose
        img_np = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img_np)
        axes[i].set_title(f'Class: {classes[labels[i]]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, classes = load_cifar10_data(batch_size=32)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {classes}")
    print(f"Number of batches (train): {len(train_loader)}")

    # Visualize some samples
    print("\nVisualizing sample images...")
    visualize_samples(train_loader, classes)
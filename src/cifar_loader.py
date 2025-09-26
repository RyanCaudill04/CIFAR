import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


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


def extract_cifar_image(index=None, dataset='test', save_path='cifar_sample.png', data_dir='./data'):
    """
    Extract a single image from CIFAR-10 dataset and save it as a file

    Args:
        index (int, optional): Index of image to extract. If None, picks random
        dataset (str): 'train' or 'test' dataset
        save_path (str): Path where to save the image
        data_dir (str): Directory containing CIFAR data

    Returns:
        tuple: (image_path, class_name, class_index)
    """

    # Load dataset without transforms to get raw data
    if dataset == 'train':
        cifar_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )
    else:
        cifar_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=None
        )

    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Select image index
    if index is None:
        index = np.random.randint(0, len(cifar_dataset))
    else:
        index = min(index, len(cifar_dataset) - 1)

    # Get the image and label
    image, label = cifar_dataset[index]

    # Convert PIL Image to numpy array and back to PIL for saving
    # CIFAR-10 images are already PIL Images
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        # Convert numpy array to PIL Image if needed
        pil_image = Image.fromarray(image)

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # Save the image
    pil_image.save(save_path)

    class_name = classes[label]

    print(f"Saved {dataset} image #{index} ({class_name}) to {save_path}")

    return save_path, class_name, label

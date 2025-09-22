# CIFAR-10 CNN Image Classifier

A PyTorch implementation of a Convolutional Neural Network (CNN) for CIFAR-10 image classification, optimized for Apple Silicon (M1/M2) MacBooks.

## Features

- **Apple Silicon Optimization**: Leverages Metal Performance Shaders (MPS) for accelerated training
- **Mixed Precision Training**: Uses autocast for optimal performance on Apple Silicon
- **Automatic Model Management**: Automatically loads the best-performing model for continued training
- **Advanced Optimization**: AdamW optimizer with OneCycleLR scheduler
- **CLI Interface**: Command-line options for flexible training workflows

## Project Structure

```
CIFAR/
├── __main__.py              # Main training script with CLI
├── models/
│   ├── cnn.py              # CNN architecture definition
│   └── *.pth               # Saved model files (cnn_accuracy.pth format)
├── src/
│   ├── cifar_loader.py     # CIFAR-10 data loading utilities
│   ├── train_model.py      # Training loop with M2 optimizations
│   └── test_model.py       # Model evaluation with M2 optimizations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CIFAR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run with default settings (loads best existing model automatically):
```bash
python __main__.py
```

### CLI Options

- `--new`: Start training with a fresh model (ignores existing saved models)
```bash
python __main__.py --new
```

- `--use <path>`: Load a specific model file to continue training from
```bash
python __main__.py --use models/cnn_85.42.pth
```

### Model Management

The system automatically:
- Saves models with accuracy in filename: `models/cnn_85.42.pth`
- Loads the highest-accuracy model by default for continued training
- Falls back to legacy naming (`simple_cnn_cifar10.pth`) for backward compatibility

## Model Architecture

The CNN consists of:
- 3 Convolutional layers (3→32→64→64 channels)
- MaxPool2d layers for downsampling
- 2 Fully connected layers (64→10 classes)
- Dropout layer (50%) for regularization
- ReLU activation functions

## Optimizations for Apple Silicon

- **MPS Backend**: Automatic detection and use of Metal Performance Shaders
- **Mixed Precision**: Autocast for optimal float16/float32 usage
- **Non-blocking Transfers**: Overlaps data movement with computation
- **AdamW Optimizer**: Superior weight decay handling
- **OneCycleLR Scheduler**: Dynamic learning rate for faster convergence

## CIFAR-10 Classes

The model classifies images into 10 categories:
- plane, car, bird, cat, deer, dog, frog, horse, ship, truck

## Performance

Optimized for MacBook Air M2:
- Faster training through MPS acceleration
- Reduced memory usage with mixed precision
- Improved convergence with advanced scheduling

## Requirements

- Python 3.8+
- PyTorch with MPS support
- torchvision
- matplotlib
- numpy

See `requirements.txt` for specific versions.

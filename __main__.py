from src import cifar_loader, train_model, test_model, cnn
import torch
import argparse
import os
import glob
import re

def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 CNN model')
    parser.add_argument('--new', action='store_true', help='Start training with a new model (ignore existing saved model)')
    parser.add_argument('--use', type=str, help='Path to existing model file to continue training from')

    args = parser.parse_args()

    print("Loading CIFAR-10 dataset...")
    trainloader, testloader, classes = cifar_loader.load_cifar10_data(16)   # Download and prepare data

    print("Creating CNN model...")
    model = cnn.cnn()                     # Create an instance of our neural network

    # Model loading logic
    model_loaded = False
    if args.use:
        # Use specific model file
        if os.path.exists(args.use):
            print(f"Loading model from {args.use}...")
            model.load_state_dict(torch.load(args.use))
            model_loaded = True
            print("Model loaded successfully!")
        else:
            print(f"Error: Model file '{args.use}' not found!")
            return
    elif not args.new:
        # Default behavior: find and load the CNN with highest accuracy
        cnn_files = glob.glob('cnn_*.pth')

        if cnn_files:
            # Extract accuracy from filename and find the highest one
            best_model = None
            best_accuracy = -1

            for file in cnn_files:
                match = re.search(r'cnn_(\d+\.\d+)\.pth', file)
                if match:
                    accuracy = float(match.group(1))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = file

            if best_model:
                print(f"Found existing model '{best_model}' with accuracy {best_accuracy:.2f}%, loading...")
                model.load_state_dict(torch.load(best_model))
                model_loaded = True
                print("Model loaded successfully!")
            else:
                print("Found CNN files but couldn't parse accuracy, starting with new model...")
        else:
            # Fallback to old naming convention
            default_model = 'simple_cnn_cifar10.pth'
            if os.path.exists(default_model):
                print(f"Found existing model '{default_model}', loading...")
                model.load_state_dict(torch.load(default_model))
                model_loaded = True
                print("Model loaded successfully!")
            else:
                print("No existing model found, starting with new model...")
    else:
        print("Starting with new model as requested...")

    # Print model architecture so we can see what we built
    print(f"\nModel Architecture:")
    print(model)                            # This shows all the layers we defined

    # Count parameters (weights and biases that the model will learn)
    total_params = sum(p.numel() for p in model.parameters())  # Count all trainable parameters
    print(f"\nTotal parameters: {total_params:,}")    # Display with comma separators

    if model_loaded:
        print("\nContinuing training from existing model...")
    else:
        print("\nStarting training with new model...")

    # Train the model and get training accuracy history
    train_accuracies = train_model.train_model(model, trainloader, testloader, num_epochs=10)

    # Final test to see how well our trained model performs
    final_accuracy = test_model.test_model(model, testloader, torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nFinal test accuracy: {final_accuracy:.2f}%")

    # Save the model so we can use it later without retraining
    model_filename = f'cnn_{final_accuracy:.2f}.pth'
    torch.save(model.state_dict(), model_filename)  # Save just the learned weights
    print(f"Model saved as '{model_filename}'")

# CIFAR-10 class names for reference - what the numbers 0-9 represent
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == "__main__":
  main()
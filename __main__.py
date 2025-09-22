from src import cifar_loader, train_model, test_model
import torch

def main():
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = cifar_loader.load_cifar10_data()   # Download and prepare data
    
    print("Creating CNN model...")
    model = model.cnn()                     # Create an instance of our neural network
    
    # Print model architecture so we can see what we built
    print(f"\nModel Architecture:")
    print(model)                            # This shows all the layers we defined
    
    # Count parameters (weights and biases that the model will learn)
    total_params = sum(p.numel() for p in model.parameters())  # Count all trainable parameters
    print(f"\nTotal parameters: {total_params:,}")    # Display with comma separators
    
    print("\nStarting training...")
    # Train the model and get training accuracy history
    train_accuracies = train_model.train_model(model, trainloader, testloader, num_epochs=5)
    
    # Final test to see how well our trained model performs
    final_accuracy = test_model.test_model(model, testloader, torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nFinal test accuracy: {final_accuracy:.2f}%")
    
    # Save the model so we can use it later without retraining
    torch.save(model.state_dict(), 'simple_cnn_cifar10.pth')  # Save just the learned weights
    print("Model saved as 'simple_cnn_cifar10.pth'")

# CIFAR-10 class names for reference - what the numbers 0-9 represent
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == 'main':
  main()
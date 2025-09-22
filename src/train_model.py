import torch                              # Main PyTorch library - gives us tensors (like arrays) and neural network basics
import torch.nn as nn                     # Neural network modules - pre-built layers like Conv2d, Linear
import torch.optim as optim              # Optimizers - algorithms that update our model's weights during training
import torch.nn.functional as F          # Functions like ReLU, softmax - operations we apply to data
import torchvision                       # Computer vision datasets and utilities
import torchvision.transforms as transforms  # Image preprocessing tools (resize, normalize, etc.)
import matplotlib.pyplot as plt          # For plotting graphs and images
import numpy as np                       # Numerical operations (like advanced calculator)

def train_model(model, trainloader, testloader, num_epochs=10):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.001)

  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    # Macbook m2 (silicon) optimization
  model.to(device)

  train_losses = []
  train_accuracies = []

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_prediction = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):
      inputs, labels = inputs.to(device), labels.to(device)     # inputs: [32, 3, 32, 32], labels: [32]

      # Reset
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      # Backward pass
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total_predictions += labels.size(0)
      correct_predictions += (predicted == labels).sum().item()

      if batch_idx % 500 == 499:      # Every 500th batch
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}], '
                      f'Loss: {running_loss/500:.4f}')  # Average loss over last 500 batches
                running_loss = 0.0

      epoch_accuracy = 100 * correct_predictions / total_predictions  # Convert to percentage
      train_accuracies.append(epoch_accuracy)  # Save for plotting later
        
        # Test the model on test data to see how well it generalizes
      test_accuracy = test_model(model, testloader, device)  # Run on test set
      print(f'Epoch [{epoch+1}/{num_epochs}] - '
            f'Train Accuracy: {epoch_accuracy:.2f}% - '
            f'Test Accuracy: {test_accuracy:.2f}%')
      
  return train_accuracies
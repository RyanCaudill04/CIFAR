import torch
import torch.nn as nn
import torch.optim as optim
from src import test_model

def train_model(model, trainloader, testloader, num_epochs=10):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005,
                                          steps_per_epoch=len(trainloader),
                                          epochs=num_epochs)

  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  model.to(device)

  # Enable mixed precision for MPS
  scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None # TODO: Optimize for MPS for back propogation

  train_losses = []
  train_accuracies = []

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):
      inputs = inputs.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)

      optimizer.zero_grad(set_to_none=True)

      # Mixed precision training
      with torch.autocast(device_type='mps' if device.type == 'mps' else 'cpu'):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

      # Backward pass with gradient scaling if using CUDA
      if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        loss.backward()
        optimizer.step()

      scheduler.step()

      running_loss += loss.item()
      predicted = outputs.argmax(dim=1)
      total_predictions += labels.size(0)
      correct_predictions += predicted.eq(labels).sum().item()

      if batch_idx % 500 == 499:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Batch [{batch_idx+1}], '
              f'Loss: {running_loss/500:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        running_loss = 0.0

    epoch_accuracy = 100 * correct_predictions / total_predictions
    train_accuracies.append(epoch_accuracy)

    # Test the model on test data to see how well it generalizes
    test_accuracy = test_model.test_model(model, testloader, device)
    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Accuracy: {epoch_accuracy:.2f}% - '
          f'Test Accuracy: {test_accuracy:.2f}%')

  return train_accuracies
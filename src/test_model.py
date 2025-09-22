import torch
import torch.nn.functional as F

def test_model(model, testloader, device=None):
  if device is None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

  model.eval()
  model = model.to(device)

  correct = 0
  total = 0

  with torch.no_grad():
    with torch.autocast(device_type='mps' if device.type == 'mps' else 'cpu'):
      for inputs, labels in testloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

  accuracy = 100 * correct / total
  return accuracy
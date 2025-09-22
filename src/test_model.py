

def test_model(model, testloader, device):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in testloader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    
  accuracy = 100 * correct / total
  return accuracy
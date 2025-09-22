import torch.nn as nn                     # Neural network modules - pre-built layers like Conv2d, Linear
import torch.nn.functional as F          # Functions like ReLU, softmax - operations we apply to data


class cnn(nn.Module):
  def __init__(self):
    super(cnn, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3,
                           out_channels=32,
                           kernel_size=3)
    
    self.conv2 = nn.Conv2d(in_channels=32,
                           out_channels=64,
                           kernel_size=3)
    
    self.conv3 = nn.Conv2d(in_channels=64,
                           out_channels=64,
                           kernel_size=3)
    
    self.pool = nn.MaxPool2d(kernel_size=2,
                             stride=2)        # 32x30x30 -> 32x15x15
    
    self.fc1 = nn.Linear(64 * 2 * 2, 64)      # Take 256 inputs, output 64 numbers
    self.fc2 = nn.Linear(64, 10)              # Take 64 inputs, output 10 numbers (one per class)

    self.dropout = nn.Dropout(0.5)            # 50% chance each neuron gets set to 0

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))

    x = x.view(-1, 64 * 2 * 2)                # -1 means "figure out this dimension automatically"

    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x
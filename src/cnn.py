import torch.nn as nn                     # Neural network modules - pre-built layers like Conv2d, Linear
import torch.nn.functional as F          # Functions like ReLU, softmax - operations we apply to data


class cnn(nn.Module):
  def __init__(self, 
                 # Input specifications
                 input_channels=3,
                 input_height=32,
                 input_width=32,
                 num_classes=10,
                 
                 # Convolutional layer specifications
                 conv1_out_channels=32,
                 conv2_out_channels=64,
                 conv3_out_channels=128,
                 conv_kernel_size=3,
                 
                 # Pooling specifications
                 pool_kernel_size=2,
                 pool_stride=2,
                 
                 # Fully connected layer specifications
                 fc1_out_features=64,
                 
                 # Regularization
                 dropout_rate=0.5):
    super(cnn, self).__init__()
        
    # Store configuration
    self.input_channels = input_channels
    self.input_height = input_height
    self.input_width = input_width
    self.num_classes = num_classes
        
    # Convolutional layer parameters
    self.conv1_out = conv1_out_channels
    self.conv2_out = conv2_out_channels
    self.conv3_out = conv3_out_channels
    self.conv_kernel = conv_kernel_size
        
    # Pooling parameters
    self.pool_kernel = pool_kernel_size
    self.pool_stride = pool_stride
        
    # FC layer parameters
    self.fc1_out = fc1_out_features
    self.dropout_rate = dropout_rate
        
    # Define convolutional layers
    self.conv1 = nn.Conv2d(in_channels=self.input_channels,
                               out_channels=self.conv1_out,
                               kernel_size=self.conv_kernel)
        
    self.conv2 = nn.Conv2d(in_channels=self.conv1_out,
                               out_channels=self.conv2_out,
                               kernel_size=self.conv_kernel)
        
    self.conv3 = nn.Conv2d(in_channels=self.conv2_out,
                               out_channels=self.conv3_out,
                               kernel_size=self.conv_kernel)
        
    # Pooling layer (shared across all conv layers)
    self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel,
                                 stride=self.pool_stride)
        
    # Batch normalization layers
    self.bn1 = nn.BatchNorm2d(self.conv1_out)
    self.bn2 = nn.BatchNorm2d(self.conv2_out)
    self.bn3 = nn.BatchNorm2d(self.conv3_out)
        
    # Calculate the flattened feature size after conv layers
    self.flattened_size = self._calculate_flattened_size()
        
    # Fully connected layers
    self.fc1 = nn.Linear(self.flattened_size, self.fc1_out)
    self.fc2 = nn.Linear(self.fc1_out, self.num_classes)
        
    # Dropout for regularization
    self.dropout = nn.Dropout(self.dropout_rate)

  def _calculate_flattened_size(self):
    """
    Calculate the size of the flattened features after all conv and pool layers.
    This helps avoid hardcoding the size.
    """
    # Track size changes through the network
    h, w = self.input_height, self.input_width
        
    # After conv1 (assuming no padding, stride=1)
    h = h - self.conv_kernel + 1
    w = w - self.conv_kernel + 1
    # After pool1
    h = h // self.pool_stride
    w = w // self.pool_stride
        
    # After conv2
    h = h - self.conv_kernel + 1
    w = w - self.conv_kernel + 1
    # After pool2
    h = h // self.pool_stride
    w = w // self.pool_stride
        
    # After conv3
    h = h - self.conv_kernel + 1
    w = w - self.conv_kernel + 1
    # After pool3
    h = h // self.pool_stride
    w = w // self.pool_stride
        
    # Total flattened size = channels * height * width
    flattened_size = self.conv3_out * h * w
        
    return flattened_size

  def forward(self, x):
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = self.pool(F.relu(self.bn3(self.conv3(x))))

    x = x.view(x.size(0), -1)                # -1 means "figure out this dimension automatically"

    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x
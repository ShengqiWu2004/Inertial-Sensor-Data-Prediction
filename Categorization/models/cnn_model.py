# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CNNModel(nn.Module):
#     def __init__(self, num_classes, window_size, input_features=6):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)

#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(128)

#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(256)

#         # Use adaptive pooling to avoid size issues
#         self.pool = nn.AdaptiveMaxPool1d(output_size=1)  # Output size is fixed to 1
#         self.dropout = nn.Dropout(0.2)
        
#         # Compute output shape dynamically
#         dummy_input = torch.zeros(1, input_features, window_size)
#         dummy_out = self._forward_conv(dummy_input)
#         self.flat_dim = dummy_out.shape[1]  # Only one dimension after flattening

#         self.fc1 = nn.Linear(self.flat_dim, 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def _forward_conv(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x)  # Adaptive pooling ensures output size is always 1
#         x = self.dropout(x)
#         x = torch.flatten(x, 1)  # Flatten along the batch dimension
#         return x

#     def forward(self, x):
#         x = x.transpose(1, 2)  # Convert shape (batch, time_steps, features) → (batch, features, time_steps)
#         x = self._forward_conv(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CNNModel(nn.Module):
#     def __init__(self, num_classes, window_size, input_features=6):
#         """
#         Args:
#             num_classes (int): Number of output classes.
#             window_size (int): Full sequence length of the input.
#             input_features (int): Number of sensor features per time step.
#         """
#         super(CNNModel, self).__init__()
#         # First convolutional block with batch normalization
#         self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.pool1 = nn.MaxPool1d(kernel_size=2)
        
#         # Second convolutional block without batch normalization
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool1d(kernel_size=2)
        
#         # Third convolutional block without batch normalization
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#         self.pool3 = nn.MaxPool1d(kernel_size=2)
        
#         # After three pooling layers (each with kernel_size=2), the time dimension is reduced by a factor of 8.
#         pooled_length = window_size // 8
        
#         self.dropout = nn.Dropout(0.2)
#         self.flat_dim = 256 * pooled_length  # 256 channels from conv3 * pooled length
#         self.fc1 = nn.Linear(self.flat_dim, 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         # x: (batch, time_steps, features) → convert to (batch, features, time_steps)
#         x = x.transpose(1, 2)
        
#         # Block 1: Convolution + BatchNorm + ReLU + MaxPooling
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool1(x)
        
#         # Block 2: Convolution + ReLU + MaxPooling
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
        
#         # Block 3: Convolution + ReLU + MaxPooling
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
        
#         x = self.dropout(x)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CNNModel(nn.Module):
#     def __init__(self, num_classes, window_size, input_features=6):
#         """
#         Args:
#             num_classes (int): Number of output classes.
#             window_size (int): Full sequence length.
#             input_features (int): Number of features per time-step.
#         """
#         super(CNNModel, self).__init__()
#         # Convolutional layers with batch normalization only in the first block.
#         self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.pool1 = nn.MaxPool1d(kernel_size=2)
        
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool1d(kernel_size=2)
        
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#         self.pool3 = nn.MaxPool1d(kernel_size=2)
        
#         # After three pooling layers, the temporal dimension is reduced by a factor of 8.
#         pooled_length = window_size // 8
        
#         self.dropout = nn.Dropout(0.2)
#         self.flat_dim = 256 * pooled_length
#         self.fc1 = nn.Linear(self.flat_dim, 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         # x: (batch, time_steps, features) → convert to (batch, features, time_steps)
#         x = x.transpose(1, 2)
        
#         # Convolution Block 1
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool1(x)
        
#         # Convolution Block 2
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
        
#         # Convolution Block 3
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
        
#         # Apply dropout
#         x = self.dropout(x)
        
#         # --- Apply Temporal Weighting Mask ---
#         # Create a mask that increases linearly from 1.0 to 1.2 over the time dimension.
#         # Note: x has shape (batch, channels, time_steps)
#         time_steps = x.shape[-1]
#         mask = torch.linspace(1.0, 1.2, steps=time_steps, device=x.device).view(1, 1, -1)
#         x = x * mask  # Broadcasting multiplies each channel and batch by the time weight.
        
#         # Flatten and pass through fully connected layers.
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes, window_size, input_features=6):
        """
        Args:
            num_classes (int): Number of output classes.
            window_size (int): Full sequence length.
            input_features (int): Number of sensor features per time-step.
        """
        super(CNNModel, self).__init__()
        # Convolutional layers with batch normalization on the first block.
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # After three pooling layers (each with kernel_size=2), the temporal dimension is reduced by factor 8.
        pooled_length = window_size // 8
        
        self.dropout = nn.Dropout(0.2)
        self.flat_dim = 256 * pooled_length
        self.fc1 = nn.Linear(self.flat_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, features) → convert to (batch, features, time_steps)
        x = x.transpose(1, 2)
        
        # Block 1: Conv + BatchNorm + ReLU + Pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 2: Conv + ReLU + Pool
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Block 3: Conv + ReLU + Pool
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = self.dropout(x)
        
        # --- Apply Temporal Weighting Mask ---
        # x shape: (batch, channels, time_steps)
        time_steps = x.shape[-1]
        # Create a mask:
        # For t < 60: weight = 1.0; for t >= 60: linearly increase from 1.0 to 1.2.
        t = torch.arange(time_steps, device=x.device).float()
        # Avoid division by zero when time_steps == 60 (if that ever occurs).
        denom = max(time_steps - 60, 1)
        weight = torch.where(t < 60, torch.ones_like(t), 1.0 + 0.2 * ((t - 60) / denom))
        # Reshape for broadcasting: (1, 1, time_steps)
        mask = weight.view(1, 1, -1)
        x = x * mask
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

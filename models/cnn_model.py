import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, window_size, predict_size, num_features=6):
        """
        CNN model for sensor data regression.
        Args:
          window_size: Number of time steps in the input sequence.
          predict_size: Number of time steps to predict.
          num_features: Number of sensor features per row (default: 6).
        """
        super(CNNModel, self).__init__()
        self.window_size = window_size
        self.predict_size = predict_size
        self.num_features = num_features
        
        # The input is (batch, window_size, num_features). For 1D conv layers, we rearrange to (batch, num_features, window_size).
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        # Adaptive pooling to convert the temporal dimension from window_size to predict_size.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.predict_size)
        
        # A final conv layer to reduce channels back to num_features.
        self.final_conv = nn.Conv1d(in_channels=64, out_channels=num_features, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
          x: Tensor of shape (batch, window_size, num_features)
          
        Returns:
          Tensor of shape (batch, predict_size, num_features)
        """
        # Permute to (batch, num_features, window_size) for Conv1d.
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Pool to get the desired temporal dimension (predict_size)
        x = self.adaptive_pool(x)  # shape: (batch, 64, predict_size)
        x = self.final_conv(x)     # shape: (batch, num_features, predict_size)
        x = x.permute(0, 2, 1)       # Back to (batch, predict_size, num_features)
        return x

if __name__ == "__main__":
    # Simple test run for the CNN model.
    window_size = 50
    predict_size = 10
    num_features = 6
    model = CNNModel(window_size, predict_size, num_features)
    dummy_input = torch.randn(8, window_size, num_features)  # batch of 8
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (8, predict_size, num_features)

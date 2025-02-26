import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNNSoftmaxModel(nn.Module):
    def __init__(self, num_classes, window_size, input_features=6):
        """
        A hybrid model that combines CNN features and basic flattened (softmax) features.
        
        Args:
            num_classes (int): Number of output classes.
            window_size (int): Full sequence length.
            input_features (int): Number of sensor features per timestep.
        """
        super(HybridCNNSoftmaxModel, self).__init__()
        # CNN Branch:
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # After 3 pooling layers, the time dimension is reduced by a factor of 8:
        pooled_length = window_size // 8
        self.cnn_feature_dim = 256 * pooled_length  # features from CNN branch
        
        # Softmax Branch (Basic flattened features)
        self.flat_feature_dim = window_size * input_features
        
        # Combined feature dimension
        self.combined_dim = self.cnn_feature_dim + self.flat_feature_dim
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.combined_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (batch, time_steps, features)
        batch_size = x.size(0)
        
        # CNN branch:
        cnn_x = x.transpose(1, 2)  # (batch, features, time_steps)
        cnn_x = F.relu(self.bn1(self.conv1(cnn_x)))
        cnn_x = self.pool1(cnn_x)
        cnn_x = F.relu(self.conv2(cnn_x))
        cnn_x = self.pool2(cnn_x)
        cnn_x = F.relu(self.conv3(cnn_x))
        cnn_x = self.pool3(cnn_x)
        cnn_x = cnn_x.view(batch_size, -1)  # flatten CNN features
        
        # Basic softmax branch: flatten the raw input
        softmax_x = x.view(batch_size, -1)
        
        # Concatenate both feature sets
        combined = torch.cat([cnn_x, softmax_x], dim=1)
        combined = self.dropout(combined)
        out = F.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class HybridCNNSoftmaxAttentionModel(nn.Module):
    def __init__(self, num_classes, window_size, input_features=6):
        """
        A hybrid model that combines CNN features (with temporal attention and weighting)
        and basic flattened raw features.

        Args:
            num_classes (int): Number of output classes.
            window_size (int): Full sequence length (number of timesteps in each segment).
            input_features (int): Number of sensor features per timestep.
        """
        super(HybridCNNSoftmaxAttentionModel, self).__init__()
        # --- CNN Branch ---
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # After three pooling layers, the time dimension is reduced by a factor of 8.
        pooled_length = window_size // 8  
        self.cnn_channels = 256
        
        # --- Temporal Weighting & Attention on CNN Branch ---
        # Apply a linear weighting to the time dimension (e.g., 1.0 to 1.2).
        # Then, compute attention weights over time.
        self.attn_fc = nn.Linear(pooled_length, pooled_length)  # simple FC to transform per-timestep scores
        
        # --- Softmax Branch ---
        # Flatten raw input (without any processing).
        self.flat_feature_dim = window_size * input_features
        
        # --- Combined Feature Processing ---
        # After applying attention, the CNN branch produces a vector of dimension (batch, 256).
        self.cnn_feature_dim = self.cnn_channels  
        self.combined_dim = self.cnn_feature_dim + self.flat_feature_dim
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.combined_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, window_size, input_features)
        batch_size = x.size(0)
        
        # --- CNN Branch ---
        cnn_x = x.transpose(1, 2)  # shape: (batch, input_features, window_size)
        cnn_x = F.relu(self.bn1(self.conv1(cnn_x)))
        cnn_x = self.pool1(cnn_x)   # shape: (batch, 64, window_size/2)
        cnn_x = F.relu(self.conv2(cnn_x))
        cnn_x = self.pool2(cnn_x)   # shape: (batch, 128, window_size/4)
        cnn_x = F.relu(self.conv3(cnn_x))
        cnn_x = self.pool3(cnn_x)   # shape: (batch, 256, pooled_length)

        # Apply temporal weighting mask: linearly increasing from 1.0 to 1.2.
        time_steps = cnn_x.shape[-1]
        mask = torch.linspace(1.0, 1.2, steps=time_steps, device=x.device).view(1, 1, -1)
        cnn_x = cnn_x * mask

        # Now apply attention over the time dimension.
        # First, compute a simple score per time step by averaging over channels.
        attn_scores = cnn_x.mean(dim=1)  # shape: (batch, pooled_length)
        # Transform these scores via a fully-connected layer.
        attn_scores = self.attn_fc(attn_scores)  # shape: (batch, pooled_length)
        # Compute attention weights with softmax.
        attn_weights = F.softmax(attn_scores, dim=1)  # shape: (batch, pooled_length)
        # Expand attention weights to match cnn_x dimensions.
        attn_weights = attn_weights.unsqueeze(1)  # shape: (batch, 1, pooled_length)
        # Aggregate CNN features using the attention weights (weighted sum over time).
        cnn_attended = (cnn_x * attn_weights).sum(dim=2)  # shape: (batch, 256)
        
        # --- Softmax Branch ---
        softmax_x = x.view(batch_size, -1)  # flatten raw input: (batch, window_size * input_features)
        
        # --- Combine Features ---
        combined = torch.cat([cnn_attended, softmax_x], dim=1)  # shape: (batch, combined_dim)
        combined = self.dropout(combined)
        out = F.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

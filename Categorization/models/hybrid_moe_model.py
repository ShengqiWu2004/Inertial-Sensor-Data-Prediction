import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridMixtureOfExpertsModel(nn.Module):
    def __init__(self, num_classes, window_size, input_features=6, num_experts=3, dropout=0.5):
        """
        A partition-based (mixture-of-experts) model that builds on the hybrid model.
        
        Args:
            num_classes (int): Number of output classes.
            window_size (int): Full sequence length (number of timesteps in each segment).
            input_features (int): Number of sensor features per timestep.
            num_experts (int): Number of expert classifiers.
            dropout (float): Dropout probability.
        """
        super(HybridMixtureOfExpertsModel, self).__init__()
        # --- Hybrid Feature Extraction ---
        # CNN Branch
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        # After 3 pooling layers, time dimension is reduced by factor of 8.
        pooled_length = window_size // 8
        self.cnn_feature_dim = 256 * pooled_length
        
        # Softmax Branch: simply flatten raw input.
        self.flat_feature_dim = window_size * input_features
        
        # Combined feature dimension.
        self.combined_dim = self.cnn_feature_dim + self.flat_feature_dim
        
        # A fully connected layer to produce a lower-dimension feature representation.
        self.dropout = nn.Dropout(dropout)
        self.fc_feature = nn.Linear(self.combined_dim, 128)
        
        # --- Mixture-of-Experts ---
        # Expert classifiers: each expert outputs logits for num_classes.
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(128, num_classes) for _ in range(num_experts)])
        # Gating network: maps features to a distribution over experts.
        self.gate = nn.Linear(128, num_experts)
        
    def forward(self, x):
        # x: (batch, time_steps, input_features)
        batch_size = x.size(0)
        # --- Hybrid Feature Extraction ---
        # CNN branch: extract features from the raw time series.
        cnn_x = x.transpose(1, 2)  # (batch, input_features, time_steps)
        cnn_x = F.relu(self.bn1(self.conv1(cnn_x)))
        cnn_x = self.pool1(cnn_x)
        cnn_x = F.relu(self.conv2(cnn_x))
        cnn_x = self.pool2(cnn_x)
        cnn_x = F.relu(self.conv3(cnn_x))
        cnn_x = self.pool3(cnn_x)
        # Apply temporal weighting mask on the CNN branch.
        time_steps = cnn_x.shape[-1]
        mask = torch.linspace(1.0, 1.2, steps=time_steps, device=x.device).view(1, 1, -1)
        cnn_x = cnn_x * mask
        cnn_x = cnn_x.view(batch_size, -1)
        
        # Softmax branch: flatten the raw input.
        softmax_x = x.view(batch_size, -1)
        
        # Concatenate the two branches.
        combined = torch.cat([cnn_x, softmax_x], dim=1)
        combined = self.dropout(combined)
        features = F.relu(self.fc_feature(combined))  # (batch, 128)
        features = self.dropout(features)
        
        # --- Mixture-of-Experts ---
        # Compute expert outputs.
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(features))  # each: (batch, num_classes)
        # Stack expert outputs: shape (batch, num_classes, num_experts)
        expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # Compute gating weights.
        gate_logits = self.gate(features)  # (batch, num_experts)
        gate_weights = F.softmax(gate_logits, dim=1)  # (batch, num_experts)
        gate_weights = gate_weights.unsqueeze(1)  # (batch, 1, num_experts)
        
        # Compute the final output as the weighted sum of expert outputs.
        output = torch.bmm(expert_outputs, gate_weights.transpose(1, 2)).squeeze(2)  # (batch, num_classes)
        return output

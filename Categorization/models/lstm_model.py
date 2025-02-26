import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_features, hidden_size=128, num_layers=2,
                 bidirectional=False, dropout=0.5, weight_threshold=60, max_weight=1.2):
        """
        Args:
            num_classes (int): Number of output classes.
            input_features (int): Number of sensor features per timestep.
            hidden_size (int): Hidden state size for LSTM.
            num_layers (int): Number of LSTM layers.
            bidirectional (bool): Use bidirectional LSTM if True.
            dropout (float): Dropout probability.
            weight_threshold (int): Time step threshold after which the weighting increases.
            max_weight (float): Maximum weight multiplier at the final time step.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.weight_threshold = weight_threshold
        self.max_weight = max_weight
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch, time_steps, input_features)
        batch_size, T, _ = x.size()
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # (batch, T, hidden_dim)
        
        # Create temporal weighting mask:
        # For t < weight_threshold, weight = 1.0;
        # for t >= weight_threshold, linearly increase from 1.0 to new_max_weight (e.g., 1.05)
        if T > self.weight_threshold:
            t = torch.arange(T, device=x.device).float()
            denom = max(T - self.weight_threshold, 1)
            new_max_weight = 1.05  # Reduced effect compared to 1.2
            mask = torch.where(t < self.weight_threshold,
                                torch.ones_like(t),
                                1.0 + (new_max_weight - 1.0) * ((t - self.weight_threshold) / denom))
        else:
            mask = torch.ones(T, device=x.device)
        mask = mask.view(1, T, 1)  # (1, T, 1)
        
        # Instead of a full weighted average, you might also consider other pooling strategies.
        weighted_out = (out * mask).sum(dim=1) / mask.sum()
        
        out = self.dropout(weighted_out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

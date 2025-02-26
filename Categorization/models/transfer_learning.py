import torch
import torch.nn as nn
import torch.nn.functional as F

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, window_size, pretrained_weights):
        super(TransferLearningModel, self).__init__()
        # Dummy pretrained base (frozen)
        self.base = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3)
        for param in self.base.parameters():
            param.requires_grad = False
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.conv = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        # Compute flattened dimension using a dummy input
        dummy_input = torch.zeros(1, window_size, 3)
        dummy_out = self._forward_conv(dummy_input)
        self.flat_dim = dummy_out.shape[1]
        self.fc1 = nn.Linear(self.flat_dim, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def _forward_conv(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.base(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

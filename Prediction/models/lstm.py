import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, predict_size=10):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_size = predict_size
        
        # LSTM: input_size=6, hidden_size=64, num_layers=2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # We'll produce predict_size * input_size outputs
        self.fc = nn.Linear(hidden_size, predict_size * input_size)
    
    def forward(self, x):
        """
        x shape: (batch, window_size, input_size=6)
        We'll produce (batch, predict_size, 6) as output.
        """
        batch_size = x.size(0)
        # LSTM
        out, (h_n, c_n) = self.lstm(x)  # out shape: (batch, window_size, hidden_size)
        
        # Take the last time step
        last_output = out[:, -1, :]  # shape: (batch, hidden_size)
        
        # Fully connected to get predict_size * input_size
        out = self.fc(last_output)  # shape: (batch, predict_size * input_size)
        
        # Reshape to (batch, predict_size, input_size)
        out = out.view(batch_size, self.predict_size, -1)
        return out

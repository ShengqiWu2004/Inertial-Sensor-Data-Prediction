import torch
import torch.nn as nn

class VanillaRNNModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, predict_size=10):
        super(VanillaRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_size = predict_size

        # Vanilla RNN: input_size=6, hidden_size=64, num_layers=2
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer to produce predict_size * input_size outputs
        self.fc = nn.Linear(hidden_size, predict_size * input_size)
    
    def forward(self, x):
        """
        x shape: (batch, window_size, input_size=6)
        Produces output of shape: (batch, predict_size, input_size)
        """
        batch_size = x.size(0)
        
        # Forward pass through the RNN
        out, h_n = self.rnn(x)  # out shape: (batch, window_size, hidden_size)
        
        # Get the output from the last time step
        last_output = out[:, -1, :]  # shape: (batch, hidden_size)
        
        # Fully connected layer maps last_output to predict_size * input_size
        out = self.fc(last_output)  # shape: (batch, predict_size * input_size)
        
        # Reshape the output to (batch, predict_size, input_size)
        out = out.view(batch_size, self.predict_size, -1)
        return out


import torch
import torch.nn as nn

class LSTMAutoregressive(nn.Module):
    """
    An improved LSTM autoregressive model that uses non-linear activation,
    proper weight initialization, and optional teacher forcing (default is 0)
    to stabilize training and prevent outputs from collapsing to zero.
    """
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, predict_size=10, teacher_forcing_ratio=0.0):
        """
        Args:
            input_size: Number of features per time step (e.g., 6 for sensor data).
            hidden_size: Dimensionality of the LSTM's hidden state.
            num_layers: Number of stacked LSTM layers.
            predict_size: Number of future time steps to predict autoregressively.
            teacher_forcing_ratio: Ratio of teacher forcing during prediction (default 0.0, meaning no teacher forcing).
        """
        super(LSTMAutoregressive, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_size = predict_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # LSTM to process the input sequence
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        # Fully connected layer that maps the hidden state to output features.
        self.fc = nn.Linear(hidden_size, input_size)
        
        # Non-linearity added after the fully connected layer.
        self.activation = nn.Tanh()
        
        # Initialize FC layer weights with Xavier initialization.
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.01)
    
    def forward(self, x, target=None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size) representing the initial sequence.
            target: Optional tensor (batch, predict_size, input_size) for teacher forcing.
        
        Returns:
            predictions: Tensor of shape (batch, predict_size, input_size) with predicted future time steps.
        """
        batch_size, seq_len, _ = x.size()
        
        # Process the input sequence
        out, (h, c) = self.lstm(x)  # out shape: (batch, seq_len, hidden_size)
        
        # Use the last hidden state to generate the first prediction
        last_output = out[:, -1, :]  # shape: (batch, hidden_size)
        current_input = self.fc(last_output)  # shape: (batch, input_size)
        current_input = self.activation(current_input)
        current_input = current_input.unsqueeze(1)  # reshape to (batch, 1, input_size)
        
        predictions = []
        h_t, c_t = h, c  # carry hidden and cell states forward
        
        # Iteratively predict for predict_size time steps
        for t in range(self.predict_size):
            out_step, (h_t, c_t) = self.lstm(current_input, (h_t, c_t))
            # out_step has shape (batch, 1, hidden_size)
            pred = self.fc(out_step[:, -1, :])  # shape: (batch, input_size)
            pred = self.activation(pred)
            pred = pred.unsqueeze(1)  # shape: (batch, 1, input_size)
            predictions.append(pred)
            
            # Optional teacher forcing: if target is provided and random chance < teacher_forcing_ratio,
            # use ground truth for the next input; otherwise, use the prediction.
            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                current_input = target[:, t:t+1, :]
            else:
                current_input = pred
        
        predictions = torch.cat(predictions, dim=1)  # shape: (batch, predict_size, input_size)
        return predictions

import torch
import torch.nn as nn

class StepwiseLSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, predict_size=10):
        super(StepwiseLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_size = predict_size
        
        # LSTM: input_size=6, hidden_size=64, num_layers=2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for output prediction
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, teacher_forcing_ratio=0.6, ground_truth=None):
        """
        x shape: (batch, window_size, input_size=6)
        We generate output step by step, predicting (batch, predict_size, 6).
        
        teacher_forcing_ratio: Probability of using ground truth instead of predicted values.
        ground_truth: Optional ground truth tensor for teacher forcing.
        """
        batch_size = x.size(0)
        h_n = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c_n = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        out, (h_n, c_n) = self.lstm(x, (h_n, c_n))  # Get the hidden state from input sequence
        
        # Use the last time step's hidden state
        decoder_input = x[:, -1, :]  # Start with the last known input (batch, input_size)
        outputs = []

        for t in range(self.predict_size):
            decoder_input = decoder_input.unsqueeze(1)  # Shape: (batch, 1, input_size)
            out, (h_n, c_n) = self.lstm(decoder_input, (h_n, c_n))  # Predict next step
            prediction = self.fc(out.squeeze(1))  # Shape: (batch, input_size)
            outputs.append(prediction)

            # Teacher forcing: use ground truth sometimes during training
            if ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = ground_truth[:, t, :]  # Use true value
            else:
                decoder_input = prediction  # Use predicted value

        return torch.stack(outputs, dim=1)  # Shape: (batch, predict_size, input_size)


import torch
import torch.nn as nn
import torch.nn.functional as F


class StrongClassifier(nn.Module):
    """
    CNN feature extractor → BiLSTM temporal encoder → attention pooling → classifier.

    CNN captures local motion patterns; BiLSTM models long-range temporal
    dependencies in both directions; attention selects the most discriminative
    timesteps for the final classification.
    """
    def __init__(self, num_classes=6, window_size=100, input_features=6,
                 cnn_filters=64, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        # --- CNN frontend: local feature extraction ---
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, cnn_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters * 2, cnn_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- BiLSTM: temporal modelling ---
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # --- Attention pooling over time ---
        self.attn = nn.Linear(lstm_hidden * 2, 1)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, time, features)
        x = x.transpose(1, 2)                          # (batch, features, time)
        x = self.cnn(x)                                 # (batch, cnn_filters*2, time)
        x = x.transpose(1, 2)                          # (batch, time, cnn_filters*2)
        x, _ = self.lstm(x)                            # (batch, time, lstm_hidden*2)
        weights = torch.softmax(self.attn(x), dim=1)  # (batch, time, 1)
        x = (x * weights).sum(dim=1)                   # (batch, lstm_hidden*2)
        return self.classifier(x)

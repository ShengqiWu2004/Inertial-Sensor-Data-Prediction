from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim=6, seq_len=15, num_classes=4):
        super(TransformerModel, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=input_dim, nhead=2)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_dim * seq_len, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)  # (batch, seq_len, input_dim)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

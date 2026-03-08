import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.pos_encoder(x)
        return self.transformer_encoder(x)

class HybridCNNTransformer(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, nhead, num_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.cnn_out_channels = cnn_out_channels
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.pool = nn.MaxPool1d(2)
        
        # After conv and pool, seq_len = input_dim // 2
        seq_len = input_dim // 2
        
        # Transformer
        self.transformer = TransformerEncoder(d_model=cnn_out_channels, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        
        # Fusion and classification head
        fusion_dim = seq_len * cnn_out_channels
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)  # (batch, cnn_out_channels, input_dim//2)
        x = x.transpose(1, 2)  # (batch, input_dim//2, cnn_out_channels)
        x = self.transformer(x)  # (batch, seq_len, cnn_out_channels)
        x = x.flatten(start_dim=1)  # (batch, seq_len * cnn_out_channels)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
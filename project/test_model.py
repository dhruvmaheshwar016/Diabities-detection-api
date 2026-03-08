import torch
from model import HybridCNNTransformer

# Test model
model = HybridCNNTransformer(
    input_dim=10,
    cnn_out_channels=64,
    nhead=8,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    num_classes=2
)

# Dummy input
x = torch.randn(32, 10)
output = model(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Model architecture:")
print(model)
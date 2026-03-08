from data_preprocessing import preprocess_data
import pandas as pd

# Test preprocessing
features, targets, scaler, encoder, selected_features = preprocess_data('data/clinical_data.csv', n_features=10)
print("Features shape:", features.shape)
print("Targets shape:", targets.shape)
print("Selected features:", selected_features.tolist())
print("Sample features:\n", features.head())
print("Sample targets:\n", targets.head())
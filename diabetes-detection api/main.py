import torch
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data
from model import HybridCNNTransformer
from train import train_model
from evaluate import evaluate_model
from explainability import explain_model

def main():
    # Paths
    data_path = 'data/clinical_data.csv'  # Update this path to your data file
    
    # Preprocess data
    features, targets, scaler, encoder, selected_features = preprocess_data(data_path, n_features=50)
    
    # Train-test split (80-20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, stratify=targets, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Data loaders
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    input_dim = X_train.shape[1]
    model = HybridCNNTransformer(
        input_dim=input_dim,
        cnn_out_channels=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=2
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model
    trained_model = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
    
    # Evaluate the model
    metrics = evaluate_model(trained_model, test_loader, device=device)
    print("Evaluation Metrics:")
    for disease, mets in metrics.items():
        print(f"{disease}: {mets}")
    
    # Explainability
    explain_model(trained_model, X_test, selected_features, device=device)

if __name__ == '__main__':
    main()
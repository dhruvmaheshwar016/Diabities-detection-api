import shap
import torch
import matplotlib.pyplot as plt
import numpy as np

def explain_model(model, X_test, feature_names, device='cpu'):
    """
    Explain the model using SHAP.

    Args:
        model (nn.Module): Trained model.
        X_test (torch.Tensor): Test features.
        feature_names (list): List of feature names.
        device (str): Device.
    """
    model.to(device)
    model.eval()
    
    # Select a subset for explanation
    X_background = X_test[:100].to(device)
    X_explain = X_test[:100].to(device)
    
    # Use DeepExplainer for PyTorch
    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(X_explain)
    
    diseases = ['diabetes', 'cad']
    
    for i, disease in enumerate(diseases):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[i], X_explain.cpu().numpy(), feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot for {disease}')
        plt.show()
        
        # Feature importance
        feature_importance = np.abs(shap_values[i]).mean(axis=0)
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importance)
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Feature Importance for {disease}')
        plt.show()
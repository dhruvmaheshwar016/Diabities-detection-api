import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the model on test data.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (str): Device.

    Returns:
        metrics (dict): Dictionary of metrics for each disease.
    """
    model.to(device)
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    diseases = ['diabetes', 'cad']
    metrics = {}
    
    plt.figure(figsize=(10, 8))
    for i, disease in enumerate(diseases):
        preds = (all_probs[:, i] > 0.5).numpy()
        targets = all_targets[:, i].numpy()
        probs = all_probs[:, i].numpy()
        
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds)
        rec = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
        auc = roc_auc_score(targets, probs)
        
        metrics[disease] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc
        }
        
        # ROC curve
        fpr, tpr, _ = roc_curve(targets, probs)
        plt.plot(fpr, tpr, label=f'{disease} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()
    
    return metrics
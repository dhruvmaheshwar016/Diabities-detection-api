"""
ROC curve generation and comparison for multiple models and diseases.

This module provides:
- Individual ROC curves per model and disease
- Comparative ROC plots
- AUC score visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns


class ROCCurveAnalyzer:
    """Generate and analyze ROC curves."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.roc_data = {}
    
    def compute_roc(self, y_true, y_proba, model_name="Model", disease_name="Disease"):
        """
        Compute ROC curve.
        
        Args:
            y_true (array-like): True labels
            y_proba (array-like): Predicted probabilities
            model_name (str): Name of model
            disease_name (str): Name of disease
            
        Returns:
            dict: FPR, TPR, thresholds, and AUC
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        key = f"{disease_name}_{model_name}"
        self.roc_data[key] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
        
        return self.roc_data[key]
    
    def plot_roc_by_disease(self, disease_name="Disease", figsize=(10, 8)):
        """
        Plot ROC curves for all models of a specific disease.
        
        Args:
            disease_name (str): Name of disease
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        for key, data in self.roc_data.items():
            if disease_name in key:
                model_name = key.replace(f"{disease_name}_", "")
                plt.plot(data['fpr'], data['tpr'], linewidth=2.5,
                        label=f"{model_name} (AUC = {data['auc']:.4f})")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curves - {disease_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_by_model(self, model_name="Model", figsize=(10, 8)):
        """
        Plot ROC curves for a specific model across diseases.
        
        Args:
            model_name (str): Name of model
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        colors = {'Diabetes': '#FF6B6B', 'CAD': '#4ECDC4'}
        
        for key, data in self.roc_data.items():
            if model_name in key:
                disease = key.replace(f"_{model_name}", "")
                color = colors.get(disease, None)
                plt.plot(data['fpr'], data['tpr'], linewidth=2.5,
                        label=f"{disease} (AUC = {data['auc']:.4f})", color=color)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_all_roc_combined(self, figsize=(16, 12)):
        """
        Plot all ROC curves in a grid.
        
        Args:
            figsize (tuple): Figure size
        """
        # Extract unique diseases and models
        diseases = set()
        models = set()
        
        for key in self.roc_data.keys():
            parts = key.split('_')
            disease = parts[0]
            model = '_'.join(parts[1:])
            diseases.add(disease)
            models.add(model)
        
        diseases = sorted(list(diseases))
        models = sorted(list(models))
        
        num_rows = len(diseases)
        num_cols = len(models)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for row, disease in enumerate(diseases):
            for col, model in enumerate(models):
                ax = axes[row, col]
                
                key = f"{disease}_{model}"
                if key in self.roc_data:
                    data = self.roc_data[key]
                    ax.plot(data['fpr'], data['tpr'], linewidth=2.5, color='#4ECDC4')
                    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
                    ax.fill_between(data['fpr'], data['tpr'], alpha=0.2, color='#4ECDC4')
                    
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1.05])
                    ax.set_title(f'{disease} - {model}\nAUC = {data["auc"]:.4f}',
                               fontsize=11, fontweight='bold')
                    ax.grid(alpha=0.3, linestyle='--')
                    
                    if col == 0:
                        ax.set_ylabel('True Positive Rate', fontsize=10)
                    else:
                        ax.set_ylabel('')
                    
                    if row == num_rows - 1:
                        ax.set_xlabel('False Positive Rate', fontsize=10)
                    else:
                        ax.set_xlabel('')
        
        plt.suptitle('ROC Curves - All Models and Diseases', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        return fig
    
    def plot_auc_comparison(self, figsize=(12, 6)):
        """
        Plot AUC scores in a comparison chart.
        
        Args:
            figsize (tuple): Figure size
        """
        # Organize data
        diseases = {}
        for key, data in self.roc_data.items():
            parts = key.split('_')
            disease = parts[0]
            model = '_'.join(parts[1:])
            
            if disease not in diseases:
                diseases[disease] = {}
            
            diseases[disease][model] = data['auc']
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=figsize)
        
        models = list(diseases[list(diseases.keys())[0]].keys())
        x = np.arange(len(models))
        width = 0.35
        
        colors = {'Diabetes': '#FF6B6B', 'CAD': '#4ECDC4'}
        
        for idx, (disease, scores) in enumerate(diseases.items()):
            values = [scores[model] for model in models]
            ax.bar(x + idx*width, values, width, label=disease,
                  color=colors.get(disease, '#45B7D1'), alpha=0.8)
        
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, 
                  label='Target (0.80)', alpha=0.7)
        
        ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
        ax.set_title('ROC-AUC Comparison Across Models and Diseases',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(models)
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for idx, disease in enumerate(diseases.keys()):
            values = [diseases[disease][model] for model in models]
            for i, v in enumerate(values):
                ax.text(i + idx*width, v + 0.02, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        return fig
    
    def get_auc_summary(self):
        """
        Get summary of all AUC scores.
        
        Returns:
            pd.DataFrame: AUC scores organized by disease and model
        """
        summary_dict = {}
        
        for key, data in self.roc_data.items():
            parts = key.split('_')
            disease = parts[0]
            model = '_'.join(parts[1:])
            
            summary_dict[key] = {'Disease': disease, 'Model': model, 'AUC': data['auc']}
        
        summary_df = pd.DataFrame(summary_dict).T
        summary_df = summary_df.sort_values('AUC', ascending=False)
        
        return summary_df
    
    def print_auc_summary(self):
        """Print AUC summary."""
        summary_df = self.get_auc_summary()
        
        print("\n" + "="*60)
        print("AUC SCORE SUMMARY")
        print("="*60)
        print(summary_df.to_string())
        print(f"\nBest Model: {summary_df.iloc[0].name} (AUC: {summary_df.iloc[0]['AUC']:.4f})")

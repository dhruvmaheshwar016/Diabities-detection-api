"""
Comprehensive model evaluation and metrics computation.

This module computes:
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC and other classification metrics
- Confusion matrices
- Per-fold stability analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, auc)
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
        self.predictions = {}
        self.comparisons = []
    
    def compute_metrics(self, y_true, y_pred, y_proba=None, model_name="Model", 
                       disease_name="Disease"):
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_proba (array-like): Predicted probabilities
            model_name (str): Name of model
            disease_name (str): Name of disease
            
        Returns:
            dict: All evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Additional metrics
        if (tp + fp) > 0:
            metrics['specificity'] = tn / (tn + fp)
        else:
            metrics['specificity'] = 0
            
        if (tp + fn) > 0:
            metrics['sensitivity'] = tp / (tp + fn)
        else:
            metrics['sensitivity'] = 0
        
        key = f"{disease_name}_{model_name}"
        self.results[key] = metrics
        
        return metrics
    
    def print_metrics(self, y_true, y_pred, y_proba=None, model_name="Model",
                     disease_name="Disease"):
        """
        Print formatted evaluation metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_proba (array-like): Predicted probabilities
            model_name (str): Name of model
            disease_name (str): Name of disease
        """
        metrics = self.compute_metrics(y_true, y_pred, y_proba, model_name, disease_name)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION: {disease_name} - {model_name}")
        print(f"{'='*60}")
        
        print(f"\n├─ Classification Metrics:")
        print(f"│  ├─ Accuracy:   {metrics['accuracy']:.4f}")
        print(f"│  ├─ Precision:  {metrics['precision']:.4f}")
        print(f"│  ├─ Recall:     {metrics['recall']:.4f}")
        print(f"│  └─ F1 Score:   {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"│")
            print(f"├─ AUC Metrics:")
            print(f"│  └─ ROC-AUC:    {metrics['roc_auc']:.4f}")
        
        print(f"│")
        print(f"├─ Sensitivity/Specificity:")
        print(f"│  ├─ Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"│  └─ Specificity: {metrics['specificity']:.4f}")
        
        print(f"│")
        print(f"└─ Confusion Matrix:")
        print(f"   ├─ True Positives:  {metrics['true_positives']}")
        print(f"   ├─ True Negatives:  {metrics['true_negatives']}")
        print(f"   ├─ False Positives: {metrics['false_positives']}")
        print(f"   └─ False Negatives: {metrics['false_negatives']}")
    
    def get_results_dataframe(self):
        """
        Get all results as DataFrame.
        
        Returns:
            pd.DataFrame: Formatted results
        """
        results_df = pd.DataFrame(self.results).T
        return results_df
    
    def compare_models(self, models_dict, disease_name="Disease"):
        """
        Compare multiple models on a dataset.
        
        Args:
            models_dict (dict): Dictionary of {model_name: model_object}
            disease_name (str): Name of disease
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison = pd.DataFrame()
        
        for model_name, model in models_dict.items():
            key = f"{disease_name}_{model_name}"
            if key in self.results:
                comparison[model_name] = pd.Series(self.results[key])
        
        return comparison.T


class PlotGenerator:
    """Generate comparison plots."""
    
    @staticmethod
    def plot_roc_curves(models_and_data, disease_name="Disease", figsize=(10, 8)):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_and_data (dict): Dictionary of {model_name: (y_true, y_proba)}
            disease_name (str): Name of disease
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        for model_name, (y_true, y_proba) in models_and_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {disease_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_model_comparison(results_dict, disease_name="Disease", figsize=(12, 6)):
        """
        Plot model performance comparison.
        
        Args:
            results_dict (dict): Dictionary of model results
            disease_name (str): Name of disease
            figsize (tuple): Figure size
        """
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize)
        
        models = list(results_dict.keys())
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [results_dict[model].get(metric, 0) for model in models]
            
            axes[idx].bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[idx].set_ylabel(metric.upper(), fontsize=11)
            axes[idx].set_ylim([0, 1])
            axes[idx].set_title(metric.upper(), fontsize=12, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        if len(axes) > 0:
            axes[0].set_xticklabels(models, rotation=45, ha='right')
            for ax in axes[1:]:
                ax.set_xticklabels([])
        
        plt.suptitle(f'Model Comparison - {disease_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name="Model", disease_name="Disease"):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            model_name (str): Name of model
            disease_name (str): Name of disease
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title(f'Confusion Matrix - {disease_name} ({model_name})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_metric_distributions(cv_results, disease_name="Disease"):
        """
        Plot distribution of metrics across CV folds.
        
        Args:
            cv_results (dict): Dictionary of {model_name: metrics_dict}
            disease_name (str): Name of disease
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        models = list(cv_results.keys())
        
        # AUC distribution
        auc_data = [cv_results[m]['fold_scores'] for m in models]
        axes[0].boxplot(auc_data, labels=models)
        axes[0].set_ylabel('ROC-AUC', fontsize=11)
        axes[0].set_title('AUC Distribution Across Folds', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Mean metrics
        metrics = ['precision', 'recall', 'f1_score']
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [cv_results[m][metric] for m in models]
            axes[1].bar(x + i*width, values, width, label=metric)
        
        axes[1].set_ylabel('Score', fontsize=11)
        axes[1].set_title('Mean Metrics', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(models)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        # Standard deviation
        std_data = [cv_results[m]['std_auc'] for m in models]
        axes[2].bar(models, std_data, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[2].set_ylabel('Standard Deviation', fontsize=11)
        axes[2].set_title('AUC Stability (Lower is Better)', fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Cross-Validation Analysis - {disease_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig

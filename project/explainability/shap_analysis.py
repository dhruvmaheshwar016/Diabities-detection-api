"""
SHAP-based explainability for model predictions.

This module provides:
- SHAP force plots for individual predictions
- Summary plots showing feature importance
- Dependence plots for feature interactions
- Model-agnostic explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')


class SHAPExplainer:
    """SHAP-based model explainability."""
    
    def __init__(self, model, X_background, feature_names=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (sklearn compatible)
            X_background (array-like): Background data for SHAP
            feature_names (list): Feature names
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_background.shape[1])]
        self.explainer = None
        self.shap_values = None
    
    def initialize(self):
        """Initialize KernelExplainer."""
        print("Initializing SHAP explainer...")
        
        # Use KernelExplainer for model-agnostic approach
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba,
            shap.sample(self.X_background, min(100, len(self.X_background)))
        )
        
        print("✓ SHAP explainer initialized")
    
    def explain_predictions(self, X_explain):
        """
        Compute SHAP values for predictions.
        
        Args:
            X_explain (array-like): Samples to explain
            
        Returns:
            np.ndarray: SHAP values
        """
        if self.explainer is None:
            self.initialize()
        
        print(f"Computing SHAP values for {len(X_explain)} samples...")
        self.shap_values = self.explainer.shap_values(X_explain)
        
        return self.shap_values
    
    def plot_summary(self, X_explain, disease_name="Disease", plot_type='bar', figsize=(12, 8)):
        """
        Plot SHAP summary plot.
        
        Args:
            X_explain (array-like): Samples to explain
            disease_name (str): Name of disease
            plot_type (str): 'bar' or 'violin'
            figsize (tuple): Figure size
        """
        if self.shap_values is None:
            self.explain_predictions(X_explain)
        
        plt.figure(figsize=figsize)
        
        # Convert to proper format for SHAP
        if isinstance(X_explain, pd.DataFrame):
            X_for_plot = X_explain.values
        else:
            X_for_plot = X_explain
        
        if plot_type == 'bar':
            # Mean absolute SHAP values
            feature_importance = np.abs(self.shap_values).mean(axis=0)
            indices = np.argsort(feature_importance)[-20:]  # Top 20
            
            sorted_features = [self.feature_names[i] for i in indices]
            sorted_importance = feature_importance[indices]
            
            plt.barh(sorted_features, sorted_importance, color='#4ECDC4')
            plt.xlabel('Mean |SHAP Value| (Average impact on model output)', fontsize=11)
            plt.title(f'Feature Importance - {disease_name}', fontsize=14, fontweight='bold')
        
        elif plot_type == 'violin':
            shap.summary_plot(self.shap_values, X_for_plot, 
                            feature_names=self.feature_names, plot_type='violin',
                            show=False)
            plt.title(f'SHAP Summary Plot - {disease_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_force(self, X_explain, sample_idx=0, disease_name="Disease"):
        """
        Plot SHAP force plot for a single prediction.
        
        Args:
            X_explain (array-like): Samples to explain
            sample_idx (int): Index of sample to explain
            disease_name (str): Name of disease
        """
        if self.shap_values is None:
            self.explain_predictions(X_explain)
        
        # Convert to proper format
        if isinstance(X_explain, pd.DataFrame):
            X_for_plot = X_explain.values
        else:
            X_for_plot = X_explain
        
        plt.figure(figsize=(14, 4))
        shap.force_plot(self.explainer.expected_value, 
                       self.shap_values[sample_idx], 
                       X_for_plot[sample_idx],
                       feature_names=self.feature_names,
                       matplotlib=True,
                       show=False)
        
        plt.title(f'SHAP Force Plot - {disease_name} (Sample {sample_idx})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_dependence(self, X_explain, feature_idx, disease_name="Disease", figsize=(10, 6)):
        """
        Plot SHAP dependence plot for a feature.
        
        Args:
            X_explain (array-like): Samples to explain
            feature_idx (int): Index of feature to analyze
            disease_name (str): Name of disease
            figsize (tuple): Figure size
        """
        if self.shap_values is None:
            self.explain_predictions(X_explain)
        
        # Convert to DataFrame if needed
        if isinstance(X_explain, pd.DataFrame):
            X_for_plot = X_explain.values
            feature_name = X_explain.columns[feature_idx] if feature_idx < len(X_explain.columns) else self.feature_names[feature_idx]
        else:
            X_for_plot = X_explain
            feature_name = self.feature_names[feature_idx]
        
        plt.figure(figsize=figsize)
        shap.dependence_plot(feature_idx, self.shap_values, X_for_plot,
                            feature_names=self.feature_names,
                            show=False)
        
        plt.title(f'SHAP Dependence Plot - {feature_name} ({disease_name})',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_feature_importance(self, top_n=20):
        """
        Get top N important features.
        
        Args:
            top_n (int): Number of top features
            
        Returns:
            pd.DataFrame: Features and their importance scores
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not yet computed")
        
        # Handle different SHAP value formats
        shap_values = self.shap_values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For binary classification, take positive class
        
        # Ensure 2D shape
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, -1]  # Take last class if 3D
        
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Ensure we have a 1D array
        if len(feature_importance.shape) > 1:
            feature_importance = feature_importance.flatten()
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def print_feature_importance(self, top_n=20):
        """Print top important features."""
        importance_df = self.get_feature_importance(top_n)
        
        print("\n" + "="*60)
        print("TOP IMPORTANT FEATURES (SHAP)")
        print("="*60)
        
        for idx, row in importance_df.iterrows():
            bar_length = int(row['Importance'] * 50)
            bar = '█' * bar_length
            print(f"{row['Feature']:25s} | {bar:50s} | {row['Importance']:.4f}")


class CompoundExplainer:
    """Combined explainability with multiple methods."""
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize compound explainer.
        
        Args:
            model: Trained model
            X_train (array-like): Training data
            feature_names (list): Feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        self.shap_explainer = None
    
    def explain_all(self, X_explain, disease_name="Disease"):
        """
        Generate comprehensive explanations.
        
        Args:
            X_explain (array-like): Samples to explain
            disease_name (str): Name of disease
        """
        print(f"\n" + "="*60)
        print(f"EXPLAINABILITY ANALYSIS: {disease_name}")
        print("="*60)
        
        # SHAP explanations - try, but gracefully handle compatibility issues
        print("\n1. SHAP Feature Importance:")
        print("-" * 60)
        
        try:
            shap_explainer = SHAPExplainer(self.model, self.X_train, self.feature_names)
            shap_explainer.explain_predictions(X_explain)
            shap_explainer.print_feature_importance(top_n=15)
        except (AttributeError, Exception) as e:
            print(f"[NOTE] SHAP initialization skipped due to model compatibility: {type(e).__name__}")
            print("Using built-in model importance instead.\n")
            shap_explainer = None
        
        # Built-in feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            print("\n2. Built-in Model Feature Importance:")
            print("-" * 60)
            
            importance = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            for idx, row in importance_df.head(15).iterrows():
                bar_length = int(row['Importance'] * 50)
                bar = '█' * bar_length
                print(f"{row['Feature']:25s} | {bar:50s} | {row['Importance']:.4f}")
        
        return shap_explainer

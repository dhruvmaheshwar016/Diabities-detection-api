"""
Comprehensive model training with cross-validation for disease prediction.

This module handles:
- Training multiple models
- 5-fold cross-validation
- Model comparison and selection
- Training metrics computation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)
import sys
sys.path.insert(0, '../models')

from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel


class MultiModelTrainer:
    """Train and compare multiple models with cross-validation."""
    
    def __init__(self, random_state=42):
        """
        Initialize trainer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
    def create_models(self):
        """Create all three models."""
        print("\n" + "="*60)
        print("INITIALIZING MODELS")
        print("="*60)
        
        self.models = {
            'Random Forest': RandomForestModel(random_state=self.random_state),
            'XGBoost': XGBoostModel(random_state=self.random_state),
            'LightGBM': LightGBMModel(random_state=self.random_state)
        }
        
        for name, model in self.models.items():
            model.build()
    
    def train_with_cv(self, X, y, cv=5, disease_name="Disease"):
        """
        Train all models with cross-validation.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (pd.Series or np.ndarray): Labels
            cv (int): Number of cross-validation folds
            disease_name (str): Name of disease being predicted
            
        Returns:
            dict: Cross-validation results
        """
        print(f"\n" + "="*60)
        print(f"TRAINING WITH {cv}-FOLD CROSS-VALIDATION: {disease_name}")
        print("="*60)
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, 
                             random_state=self.random_state)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'─'*60}")
            print(f"Model: {model_name}")
            print(f"{'─'*60}")
            
            fold_scores_train = []
            fold_scores_val = []
            fold_auc = []
            fold_predictions = []
            fold_probas = []
            fold_true_labels = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_train_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
                X_val_fold = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
                y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
                y_val_fold = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
                
                # Create fresh model instance for each fold
                if model_name == 'Random Forest':
                    fold_model = RandomForestModel(random_state=self.random_state + fold)
                elif model_name == 'XGBoost':
                    fold_model = XGBoostModel(random_state=self.random_state + fold)
                else:  # LightGBM
                    fold_model = LightGBMModel(random_state=self.random_state + fold)
                
                fold_model.build()
                fold_model.train(X_train_fold, y_train_fold, verbose=False)
                
                # Evaluate
                train_acc = fold_model.model.score(X_train_fold, y_train_fold)
                val_acc = fold_model.model.score(X_val_fold, y_val_fold)
                
                y_val_pred = fold_model.predict(X_val_fold)
                y_val_proba = fold_model.predict_proba(X_val_fold)
                
                auc = roc_auc_score(y_val_fold, y_val_proba)
                
                fold_scores_train.append(train_acc)
                fold_scores_val.append(val_acc)
                fold_auc.append(auc)
                fold_predictions.append(y_val_pred)
                fold_probas.append(y_val_proba)
                fold_true_labels.append(y_val_fold.values if isinstance(y_val_fold, pd.Series) 
                                       else y_val_fold)
                
                print(f"  Fold {fold}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, AUC={auc:.4f}")
            
            # Aggregate results
            mean_train_acc = np.mean(fold_scores_train)
            mean_val_acc = np.mean(fold_scores_val)
            mean_auc = np.mean(fold_auc)
            std_auc = np.std(fold_auc)
            
            # Combine predictions from all folds
            all_true = np.concatenate(fold_true_labels)
            all_pred = np.concatenate(fold_predictions)
            all_proba = np.concatenate(fold_probas)
            
            precision = precision_score(all_true, all_pred, zero_division=0)
            recall = recall_score(all_true, all_pred, zero_division=0)
            f1 = f1_score(all_true, all_pred, zero_division=0)
            
            cv_results[model_name] = {
                'train_accuracy': mean_train_acc,
                'val_accuracy': mean_val_acc,
                'mean_auc': mean_auc,
                'std_auc': std_auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fold_scores': fold_scores_val
            }
            
            print(f"\n  ╔═ Cross-Validation Summary ═╗")
            print(f"  ║ Mean Train Acc: {mean_train_acc:.4f}")
            print(f"  ║ Mean Val Acc:   {mean_val_acc:.4f}")
            print(f"  ║ Mean AUC:       {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"  ║ Precision:      {precision:.4f}")
            print(f"  ║ Recall:         {recall:.4f}")
            print(f"  ║ F1 Score:       {f1:.4f}")
            print(f"  ╚════════════════════════════════╝")
        
        self.cv_results[disease_name] = cv_results
        return cv_results
    
    def train_final_models(self, X_train, y_train, disease_name="Disease"):
        """
        Train final models on full training set.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (pd.Series or np.ndarray): Training labels
            disease_name (str): Name of disease being predicted
        """
        print(f"\n" + "="*60)
        print(f"TRAINING FINAL MODELS: {disease_name}")
        print("="*60)
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model.train(X_train, y_train, verbose=True)
            trained_models[model_name] = model
        
        if disease_name not in self.trained_models:
            self.trained_models[disease_name] = {}
        
        self.trained_models[disease_name].update(trained_models)
    
    def select_best_model(self, disease_name="Disease", metric='mean_auc'):
        """
        Select best model based on CV results.
        
        Args:
            disease_name (str): Name of disease
            metric (str): Metric to use for selection
            
        Returns:
            tuple: (best_model_name, best_score)
        """
        if disease_name not in self.cv_results:
            raise ValueError(f"No CV results for {disease_name}")
        
        results = self.cv_results[disease_name]
        best_model = max(results.items(), key=lambda x: x[1][metric])
        
        self.best_model_name = best_model[0]
        self.best_model = self.trained_models[disease_name][best_model[0]]
        
        print(f"\n✓ Best model selected: {best_model[0]}")
        print(f"  {metric.upper()}: {best_model[1][metric]:.4f}")
        
        return best_model[0], best_model[1][metric]
    
    def get_results_dataframe(self):
        """
        Get all CV results as DataFrame.
        
        Returns:
            dict: DataFrames of results for each disease
        """
        results_dfs = {}
        
        for disease_name, cv_results in self.cv_results.items():
            results_df = pd.DataFrame(cv_results).T
            results_df = results_df[['train_accuracy', 'val_accuracy', 'mean_auc', 
                                     'std_auc', 'precision', 'recall', 'f1_score']]
            results_dfs[disease_name] = results_df
        
        return results_dfs
    
    def print_comparison(self):
        """Print model comparison summary."""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for disease_name, results_df in self.get_results_dataframe().items():
            print(f"\n{disease_name}:")
            print(results_df.to_string())


# Convenience function
def train_models(X_diabetes, y_diabetes, X_cad, y_cad, cv=5):
    """
    Convenience function to train all models for both diseases.
    
    Args:
        X_diabetes (pd.DataFrame): Features for diabetes
        y_diabetes (pd.Series): Labels for diabetes
        X_cad (pd.DataFrame): Features for CAD
        y_cad (pd.Series): Labels for CAD
        cv (int): Number of CV folds
        
    Returns:
        MultiModelTrainer: Trained trainer object
    """
    trainer = MultiModelTrainer()
    trainer.create_models()
    
    # Train for diabetes
    trainer.train_with_cv(X_diabetes, y_diabetes, cv=cv, disease_name='Diabetes')
    trainer.train_final_models(X_diabetes, y_diabetes, disease_name='Diabetes')
    trainer.select_best_model('Diabetes')
    
    # Train for CAD
    trainer.train_with_cv(X_cad, y_cad, cv=cv, disease_name='CAD')
    trainer.train_final_models(X_cad, y_cad, disease_name='CAD')
    trainer.select_best_model('CAD')
    
    # Summary
    trainer.print_comparison()
    
    return trainer

"""
LightGBM model for medical disease prediction.

LightGBM (Light Gradient Boosting Machine) is:
- A fast and efficient gradient boosting framework
- Uses leaf-wise tree growth
- Memory efficient and scalable
- Particularly effective for large datasets with many features
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import pickle


class LightGBMModel:
    """LightGBM classifier for disease prediction."""
    
    def __init__(self, n_estimators=200, max_depth=7, learning_rate=0.05,
                 num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                 min_data_in_leaf=20, random_state=42, n_jobs=-1):
        """
        Initialize LightGBM model.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate
            num_leaves (int): Number of leaves in tree
            subsample (float): Subsample ratio of training instances
            colsample_bytree (float): Subsample ratio of columns
            min_data_in_leaf (int): Minimum data points in leaf
            random_state (int): Random seed
            n_jobs (int): Number of parallel threads
        """
        self.hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_data_in_leaf': min_data_in_leaf,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'objective': 'binary',
            'metric': 'binary_logloss'
        }
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        
    def build(self):
        """Build the LightGBM model."""
        self.model = lgb.LGBMClassifier(**self.hyperparameters)
        print("✓ LightGBM model built")
        
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the LightGBM model.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (pd.Series or np.ndarray): Training labels
            X_val (pd.DataFrame or np.ndarray): Validation features
            y_val (pd.Series or np.ndarray): Validation labels
            verbose (bool): Print training info
            
        Returns:
            dict: Training metrics
        """
        if self.model is None:
            self.build()
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        fit_kwargs = {}
        if X_val is not None and eval_set:
            fit_kwargs['eval_set'] = eval_set
        
        self.model.fit(X_train, y_train, **fit_kwargs)
        
        self.feature_importance = self.model.feature_importances_
        self.is_trained = True
        
        train_score = self.model.score(X_train, y_train)
        
        if verbose:
            print(f"✓ LightGBM trained")
            print(f"  Training accuracy: {train_score:.4f}")
            if X_val is not None:
                val_score = self.model.score(X_val, y_val)
                print(f"  Validation accuracy: {val_score:.4f}")
        
        return {'train_accuracy': train_score}
    
    def predict(self, X):
        """
        Make binary predictions.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            
        Returns:
            np.ndarray: Predicted class labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            
        Returns:
            np.ndarray: Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)[:, 1]
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (pd.Series or np.ndarray): Labels
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        if self.model is None:
            self.build()
        
        scores = cross_val_score(self.model, X, y, cv=cv,
                                scoring='roc_auc', n_jobs=-1)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def get_feature_importance(self, feature_names=None, top_n=20, importance_type='split'):
        """
        Get feature importance scores.
        
        Args:
            feature_names (list): Feature names
            top_n (int): Number of top features to return
            importance_type (str): 'split' or 'gain'
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, filepath):
        """Save model to file."""
        self.model.booster_.save_model(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        if self.model is None:
            self.build()
        booster = lgb.Booster(model_file=filepath)
        self.model = lgb.LGBMClassifier()
        self.model.booster_ = booster
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")

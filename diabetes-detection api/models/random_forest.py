"""
Random Forest model for medical disease prediction.

Random Forest is an ensemble method that:
- Creates multiple decision trees
- Reduces overfitting through averaging
- Handles non-linear relationships well
- Provides feature importance scores
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle


class RandomForestModel:
    """Random Forest classifier for disease prediction."""
    
    def __init__(self, n_estimators=200, max_depth=20, min_samples_split=5,
                 min_samples_leaf=2, random_state=42, n_jobs=-1):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            min_samples_split (int): Min samples to split
            min_samples_leaf (int): Min samples in leaf
            random_state (int): Random seed
            n_jobs (int): Number of parallel jobs
        """
        self.hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        
    def build(self):
        """Build the Random Forest model."""
        self.model = RandomForestClassifier(**self.hyperparameters)
        print("✓ Random Forest model built")
        
    def train(self, X_train, y_train, verbose=True):
        """
        Train the Random Forest model.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (pd.Series or np.ndarray): Training labels
            verbose (bool): Print training info
            
        Returns:
            dict: Training metrics
        """
        if self.model is None:
            self.build()
        
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
        self.is_trained = True
        
        train_score = self.model.score(X_train, y_train)
        
        if verbose:
            print(f"✓ Random Forest trained")
            print(f"  Training accuracy: {train_score:.4f}")
        
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
    
    def get_feature_importance(self, feature_names=None, top_n=20):
        """
        Get feature importance scores.
        
        Args:
            feature_names (list): Feature names
            top_n (int): Number of top features to return
            
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
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")

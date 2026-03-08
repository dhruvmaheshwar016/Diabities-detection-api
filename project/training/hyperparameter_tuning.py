"""
Hyperparameter tuning for disease prediction models.

This module uses GridSearchCV and RandomizedSearchCV for:
- Systematic hyperparameter optimization
- Cross-validated parameter selection
- Best parameter identification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """Optimize hyperparameters for multiple models."""
    
    def __init__(self, cv=5, random_state=42, n_jobs=-1):
        """
        Initialize optimizer.
        
        Args:
            cv (int): Number of cross-validation folds
            random_state (int): Random seed
            n_jobs (int): Number of parallel jobs
        """
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params = {}
        self.search_results = {}
    
    def optimize_random_forest(self, X, y, search_type='random', n_iter=20):
        """
        Optimize Random Forest hyperparameters.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            search_type (str): 'grid' or 'random'
            n_iter (int): Number of iterations for random search
            
        Returns:
            dict: Best parameters
        """
        print("\n" + "="*60)
        print("OPTIMIZING RANDOM FOREST")
        print("="*60)
        
        param_grid = {
            'n_estimators': [100, 150, 200, 300],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        model = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=self.cv, 
                                scoring='roc_auc', n_jobs=self.n_jobs, 
                                verbose=1)
        else:
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter,
                                       cv=self.cv, scoring='roc_auc',
                                       n_jobs=self.n_jobs, random_state=self.random_state,
                                       verbose=1)
        
        search.fit(X, y)
        
        best_params = search.best_params_
        best_score = search.best_score_
        
        self.best_params['Random Forest'] = best_params
        self.search_results['Random Forest'] = search
        
        print(f"\n✓ Best Parameters (AUC: {best_score:.4f}):")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        return best_params
    
    def optimize_xgboost(self, X, y, search_type='random', n_iter=20):
        """
        Optimize XGBoost hyperparameters.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            search_type (str): 'grid' or 'random'
            n_iter (int): Number of iterations for random search
            
        Returns:
            dict: Best parameters
        """
        print("\n" + "="*60)
        print("OPTIMIZING XGBOOST")
        print("="*60)
        
        param_grid = {
            'n_estimators': [100, 150, 200, 300],
            'max_depth': [5, 7, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        model = xgb.XGBClassifier(objective='binary:logistic', random_state=self.random_state,
                                 eval_metric='logloss', use_label_encoder=False)
        
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=self.cv,
                                scoring='roc_auc', n_jobs=self.n_jobs,
                                verbose=1)
        else:
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter,
                                       cv=self.cv, scoring='roc_auc',
                                       n_jobs=self.n_jobs, random_state=self.random_state,
                                       verbose=1)
        
        search.fit(X, y)
        
        best_params = search.best_params_
        best_score = search.best_score_
        
        self.best_params['XGBoost'] = best_params
        self.search_results['XGBoost'] = search
        
        print(f"\n✓ Best Parameters (AUC: {best_score:.4f}):")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        return best_params
    
    def optimize_lightgbm(self, X, y, search_type='random', n_iter=20):
        """
        Optimize LightGBM hyperparameters.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            search_type (str): 'grid' or 'random'
            n_iter (int): Number of iterations for random search
            
        Returns:
            dict: Best parameters
        """
        print("\n" + "="*60)
        print("OPTIMIZING LIGHTGBM")
        print("="*60)
        
        param_grid = {
            'n_estimators': [100, 150, 200, 300],
            'max_depth': [5, 7, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 31, 50, 100],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_data_in_leaf': [10, 20, 30]
        }
        
        model = lgb.LGBMClassifier(objective='binary', random_state=self.random_state)
        
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=self.cv,
                                scoring='roc_auc', n_jobs=self.n_jobs,
                                verbose=1)
        else:
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter,
                                       cv=self.cv, scoring='roc_auc',
                                       n_jobs=self.n_jobs, random_state=self.random_state,
                                       verbose=1)
        
        search.fit(X, y)
        
        best_params = search.best_params_
        best_score = search.best_score_
        
        self.best_params['LightGBM'] = best_params
        self.search_results['LightGBM'] = search
        
        print(f"\n✓ Best Parameters (AUC: {best_score:.4f}):")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        return best_params
    
    def optimize_all(self, X, y, search_type='random', n_iter=20):
        """
        Optimize all models.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            search_type (str): 'grid' or 'random'
            n_iter (int): Number of iterations for random search
            
        Returns:
            dict: Best parameters for all models
        """
        self.optimize_random_forest(X, y, search_type=search_type, n_iter=n_iter)
        self.optimize_xgboost(X, y, search_type=search_type, n_iter=n_iter)
        self.optimize_lightgbm(X, y, search_type=search_type, n_iter=n_iter)
        
        return self.best_params
    
    def get_best_params(self, model_name):
        """
        Get best parameters for a specific model.
        
        Args:
            model_name (str): Name of model
            
        Returns:
            dict: Best parameters
        """
        return self.best_params.get(model_name, {})
    
    def get_optimization_history(self, model_name):
        """
        Get optimization history for a model.
        
        Args:
            model_name (str): Name of model
            
        Returns:
            pd.DataFrame: Results from search
        """
        if model_name not in self.search_results:
            return None
        
        search = self.search_results[model_name]
        results_df = pd.DataFrame(search.cv_results_)
        
        return results_df[['param_' + p for p in search.param_grid.keys()] + 
                         ['mean_test_score', 'std_test_score', 'rank_test_score']]
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        
        for model_name in self.best_params.keys():
            print(f"\n{model_name}:")
            for param, value in self.best_params[model_name].items():
                print(f"  {param}: {value}")

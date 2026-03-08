"""
Specialized diabetes prediction optimization module.

Focuses on improving diabetes prediction through:
- Feature engineering for diabetes-specific predictors
- Class imbalance handling with SMOTE
- Hyperparameter tuning with RandomizedSearchCV
- 5-fold cross-validation
- Comprehensive model comparison and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (roc_auc_score, roc_curve, auc, accuracy_score, 
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
matplotlib_backend = plt.matplotlib
if hasattr(matplotlib_backend, 'use'):
    matplotlib_backend.use('Agg')


class DiabetesDataPreprocessor:
    """Specialized preprocessor for diabetes prediction."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
    def load_and_process(self, data_path):
        """Load and process data for diabetes prediction."""
        print("\n" + "="*70)
        print("DIABETES-SPECIFIC DATA PREPROCESSING")
        print("="*70)
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"[OK] Original dataset: {df.shape}")
        
        # Separate features and target
        X = df.drop(['diabetes', 'cad'], axis=1)
        y = df['diabetes'].copy()
        
        print(f"\n[INFO] Diabetes class distribution:")
        print(f"  Class 0 (No):  {(y==0).sum()} samples ({(y==0).sum()/len(y)*100:.1f}%)")
        print(f"  Class 1 (Yes): {(y==1).sum()} samples ({(y==1).sum()/len(y)*100:.1f}%)")
        print(f"  Imbalance Ratio: {(y==0).sum() / (y==1).sum():.2f}:1")
        
        # Engineer diabetes-specific features (before encoding)
        X = self._engineer_features(X)
        
        # Encode categorical features BEFORE imputation
        X = self._encode_categoricals(X)
        
        # Handle missing values
        print(f"\n[INFO] Missing values before imputation: {X.isnull().sum().sum()}")
        X = self._impute_missing(X)
        print(f"[OK] Missing values after imputation: {X.isnull().sum().sum()}")
        
        # Remove highly correlated features
        X = self._remove_correlated_features(X, threshold=0.95)
        
        # Scale features
        X = self._scale_features(X)
        
        self.feature_names = X.columns.tolist()
        
        print(f"\n[OK] Final feature set: {len(self.feature_names)} features")
        print(f"[OK] Selected features: {', '.join(self.feature_names)}")
        
        return X, y
    
    def _engineer_features(self, X):
        """Engineer diabetes-specific features."""
        print("\n[INFO] Engineering diabetes-specific features...")
        
        X_eng = X.copy()
        
        # BMI-based features
        if 'bmi' in X_eng.columns:
            X_eng['bmi_squared'] = X_eng['bmi'] ** 2
            X_eng['bmi_category'] = pd.cut(X_eng['bmi'], 
                                          bins=[0, 18.5, 25, 30, 100],
                                          labels=[0, 1, 2, 3]).astype(float)
        
        # Glucose-related features
        if 'glucose' in X_eng.columns:
            X_eng['glucose_high'] = (X_eng['glucose'] > 126).astype(int)
            X_eng['glucose_fasting_risk'] = (X_eng['glucose'] > 100).astype(int)
        
        # Insulin resistance proxy (HOMA-IR approximation)
        if 'glucose' in X_eng.columns and 'insulin' in X_eng.columns:
            # HOMA-IR = (Fasting Glucose * Fasting Insulin) / 405
            X_eng['homa_ir'] = (X_eng['glucose'] * X_eng['insulin']) / 405.0
            X_eng['homa_ir_high'] = (X_eng['homa_ir'] > 2.5).astype(int)
        
        # Lipid ratios
        if 'triglycerides' in X_eng.columns and 'hdl' in X_eng.columns:
            X_eng['triglyceride_hdl_ratio'] = X_eng['triglycerides'] / (X_eng['hdl'] + 1e-6)
        
        if 'ldl' in X_eng.columns and 'hdl' in X_eng.columns:
            X_eng['ldl_hdl_ratio'] = X_eng['ldl'] / (X_eng['hdl'] + 1e-6)
        
        # Age-related risk
        if 'age' in X_eng.columns:
            X_eng['age_risk'] = pd.cut(X_eng['age'], 
                                       bins=[0, 35, 50, 65, 150],
                                       labels=[0, 1, 2, 3]).astype(float)
        
        # Metabolic risk score (composite)
        risk_factors = []
        if 'glucose' in X_eng.columns:
            risk_factors.append((X_eng['glucose'] > 100).astype(int))
        if 'bmi' in X_eng.columns:
            risk_factors.append((X_eng['bmi'] > 25).astype(int))
        if 'blood_pressure' in X_eng.columns:
            risk_factors.append((X_eng['blood_pressure'] > 130).astype(int))
        
        if risk_factors:
            X_eng['metabolic_risk_score'] = sum(risk_factors)
        
        print(f"[OK] Engineered {len(X_eng.columns) - len(X.columns)} new features")
        
        return X_eng
    
    def _impute_missing(self, X):
        """Impute missing values using KNN."""
        self.imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )
        return X_imputed
    
    def _encode_categoricals(self, X):
        """Encode categorical features."""
        X_enc = X.copy()
        
        # Map gender
        if 'gender' in X_enc.columns:
            gender_map = {'Male': 1, 'Female': 0}
            X_enc['gender'] = X_enc['gender'].map(gender_map)
        
        # Map smoking
        if 'smoking' in X_enc.columns:
            smoking_map = {'Yes': 1, 'No': 0}
            X_enc['smoking'] = X_enc['smoking'].map(smoking_map)
        
        # Map family_history
        if 'family_history' in X_enc.columns:
            fh_map = {'Yes': 1, 'No': 0}
            X_enc['family_history'] = X_enc['family_history'].map(fh_map)
        
        return X_enc
    
    def _remove_correlated_features(self, X, threshold=0.95):
        """Remove highly correlated features."""
        print(f"\n[INFO] Removing highly correlated features (threshold={threshold})...")
        
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        drop_features = [column for column in upper.columns 
                        if any(upper[column] > threshold)]
        
        if drop_features:
            print(f"[OK] Dropped {len(drop_features)} correlated features: {drop_features}")
            X_filtered = X.drop(columns=drop_features)
        else:
            print(f"[OK] No highly correlated features to remove")
            X_filtered = X
        
        return X_filtered
    
    def _scale_features(self, X):
        """Scale features using StandardScaler."""
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        return X_scaled


class DiabetesModelOptimizer:
    """Optimize diabetes prediction models."""
    
    def __init__(self, random_state=42, cv=5):
        self.random_state = random_state
        self.cv = cv
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def create_models(self):
        """Create baseline models."""
        print("\n" + "="*70)
        print("CREATING BASELINE MODELS")
        print("="*70)
        
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method='hist',
                eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                is_unbalanced=True
            )
        }
        
        for name in self.models.keys():
            print(f"[OK] Created {name} model")
    
    def apply_smote_and_train(self, X, y):
        """Apply SMOTE and train models with hyperparameter tuning."""
        print("\n" + "="*70)
        print("SMOTE OVERSAMPLING AND HYPERPARAMETER TUNING")
        print("="*70)
        
        # Apply SMOTE to handle class imbalance
        print(f"\n[INFO] Original class distribution:")
        print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"[OK] After SMOTE:")
        print(f"  Class 0: {(y_resampled==0).sum()}, Class 1: {(y_resampled==1).sum()}")
        
        # Define hyperparameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'XGBoost': {
                'max_depth': [5, 7, 9, 11],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'lambda': [0.1, 1, 10]
            },
            'LightGBM': {
                'num_leaves': [15, 31, 63],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_samples': [10, 20, 30],
                'lambda_l1': [0, 0.1, 1]
            }
        }
        
        # Hyperparameter tuning with RandomizedSearchCV
        for model_name, model in self.models.items():
            print(f"\n[INFO] Tuning {model_name} hyperparameters...")
            
            param_grid = param_grids[model_name]
            
            random_search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=20,
                cv=min(3, self.cv),
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            
            random_search.fit(X_resampled, y_resampled)
            
            self.models[model_name] = random_search.best_estimator_
            
            print(f"[OK] {model_name} best CV score: {random_search.best_score_:.4f}")
            print(f"[OK] Best parameters: {random_search.best_params_}")
    
    def cross_validate_models(self, X, y):
        """Perform 5-fold cross-validation."""
        print("\n" + "="*70)
        print("5-FOLD STRATIFIED CROSS-VALIDATION")
        print("="*70)
        
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, 
                             random_state=self.random_state)
        
        results_list = []
        
        for model_name, model in self.models.items():
            print(f"\n[INFO] Cross-validating {model_name}...")
            
            fold_results = []
            fold_num = 1
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Apply SMOTE to training fold only
                smote = SMOTE(random_state=self.random_state)
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
                
                # Train model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train_sm, y_train_sm)
                
                # Evaluate
                y_pred = model_copy.predict(X_test)
                y_pred_proba = model_copy.predict_proba(X_test)[:, 1]
                
                auc_score = roc_auc_score(y_test, y_pred_proba)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                fold_results.append({
                    'Fold': fold_num,
                    'AUC': auc_score,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1
                })
                
                print(f"  Fold {fold_num}: AUC={auc_score:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
                
                fold_num += 1
            
            # Aggregate results
            fold_df = pd.DataFrame(fold_results)
            mean_results = fold_df[['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']].mean()
            std_results = fold_df[['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']].std()
            
            self.results[model_name] = {
                'means': mean_results,
                'stds': std_results,
                'fold_details': fold_df
            }
            
            print(f"[OK] {model_name} Mean AUC: {mean_results['AUC']:.4f} +/- {std_results['AUC']:.4f}")
            
            # Track best model
            if mean_results['AUC'] > self.best_score:
                self.best_score = mean_results['AUC']
                self.best_model = model_name
    
    def _clone_model(self, model):
        """Clone a model for cross-validation."""
        if isinstance(model, RandomForestClassifier):
            return RandomForestClassifier(**model.get_params())
        elif isinstance(model, XGBClassifier):
            return XGBClassifier(**model.get_params())
        elif isinstance(model, LGBMClassifier):
            return LGBMClassifier(**model.get_params())
    
    def train_final_models(self, X, y):
        """Train final models on full dataset."""
        print("\n" + "="*70)
        print("TRAINING FINAL MODELS")
        print("="*70)
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        self.final_models = {}
        for model_name, model in self.models.items():
            print(f"\n[INFO] Training final {model_name} on full dataset...")
            
            model_copy = self._clone_model(model)
            model_copy.fit(X_resampled, y_resampled)
            self.final_models[model_name] = model_copy
            
            # Train and validation metrics
            y_train_pred_proba = model_copy.predict_proba(X)[:, 1]
            train_auc = roc_auc_score(y, y_train_pred_proba)
            
            print(f"[OK] {model_name} trained. Train AUC: {train_auc:.4f}")
    
    def get_comparison_dataframe(self):
        """Get model comparison results as DataFrame."""
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'AUC': result['means']['AUC'],
                'AUC_std': result['stds']['AUC'],
                'Accuracy': result['means']['Accuracy'],
                'Precision': result['means']['Precision'],
                'Recall': result['means']['Recall'],
                'F1': result['means']['F1']
            })
        
        return pd.DataFrame(comparison_data).sort_values('AUC', ascending=False)
    
    def print_results_summary(self):
        """Print comprehensive results summary."""
        print("\n" + "="*70)
        print("DIABETES PREDICTION - RESULTS SUMMARY")
        print("="*70)
        
        comparison_df = self.get_comparison_dataframe()
        
        print("\n1. CROSS-VALIDATION RESULTS:")
        print("-" * 70)
        print(comparison_df.to_string(index=False))
        
        print(f"\n2. BEST MODEL SELECTED:")
        print("-" * 70)
        print(f"Model: {self.best_model}")
        print(f"AUC: {self.best_score:.4f}")
        
        if self.best_score > 0.85:
            print(f"[SUCCESS] Target AUC > 0.85 achieved!")
        else:
            print(f"[INFO] Current AUC: {self.best_score:.4f}, Target: 0.85")
        
        return comparison_df


class DiabetesVisualization:
    """Generate diabetes prediction visualizations."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        self.roc_data = {}
    
    def plot_roc_curves(self, X_test, y_test, models_dict, model_names, feature_names=None):
        """Plot ROC curves for all models."""
        print(f"\n[INFO] Generating ROC curves...")
        
        plt.figure(figsize=(12, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[idx], lw=2.5,
                    label=f'{model_name} (AUC = {auc_score:.4f})')
            
            self.roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5000)')
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Diabetes Prediction (Final Models)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/diabetes_roc_curves.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {filepath}")
    
    def plot_model_comparison(self, comparison_df):
        """Plot model comparison metrics."""
        print(f"[INFO] Generating model comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['AUC', 'Accuracy', 'Precision', 'Recall']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(comparison_df)],
                          edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        filepath = f'{self.output_dir}/diabetes_model_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {filepath}")
    
    def plot_feature_importance(self, model, feature_names, model_name='Best Model'):
        """Plot feature importance."""
        print(f"[INFO] Generating feature importance plot for {model_name}...")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Trim to top 15
            indices = np.argsort(importances)[::-1][:15]
            
            plt.figure(figsize=(10, 8))
            
            plt.barh(range(len(indices)), importances[indices],
                    color='#45B7D1', edgecolor='black', linewidth=1.5)
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
            plt.title(f'Top 15 Feature Importances - {model_name}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            filepath = f'{self.output_dir}/diabetes_feature_importance.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Saved: {filepath}")


def main():
    """Main diabetes optimization pipeline."""
    print("\n" + "="*70)
    print("DIABETES PREDICTION OPTIMIZATION PIPELINE")
    print("="*70)
    
    # 1. Data preprocessing
    preprocessor = DiabetesDataPreprocessor(random_state=42)
    X, y = preprocessor.load_and_process('data/clinical_data.csv')
    
    # Create train/test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Create and optimize models
    optimizer = DiabetesModelOptimizer(random_state=42, cv=5)
    optimizer.create_models()
    optimizer.apply_smote_and_train(X_train, y_train)
    optimizer.cross_validate_models(X_train, y_train)
    optimizer.train_final_models(X_train, y_train)
    
    # 3. Print results
    comparison_df = optimizer.print_results_summary()
    
    # 4. Generate visualizations
    viz = DiabetesVisualization(output_dir='results')
    viz.plot_roc_curves(X_test, y_test, optimizer.final_models, 
                        list(optimizer.final_models.keys()),
                        preprocessor.feature_names)
    viz.plot_model_comparison(comparison_df)
    
    if optimizer.best_model:
        viz.plot_feature_importance(
            optimizer.final_models[optimizer.best_model],
            preprocessor.feature_names,
            optimizer.best_model
        )
    
    # 5. Save best model
    if optimizer.best_model:
        filepath = f'results/diabetes_best_model_{optimizer.best_model}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(optimizer.final_models[optimizer.best_model], f)
        print(f"\n[OK] Best model saved: {filepath}")
    
    print("\n" + "="*70)
    print("DIABETES OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

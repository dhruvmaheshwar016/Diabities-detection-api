"""
Enhanced Diabetes Prediction Pipeline with SMOTE and Advanced Feature Engineering.

This pipeline:
1. Applies SMOTE oversampling to handle class imbalance (2.4:1 ratio)
2. Creates diabetes-specific engineered features
3. Trains multiple models with hyperparameter tuning
4. Implements 5-fold cross-validation
5. Evaluates models with ROC curves and metrics
6. Saves the best model

Target: Achieve ROC-AUC > 0.85 for diabetes prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    RandomizedSearchCV, cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix, 
    classification_report, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import warnings
import pickle
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DiabetesEnhancedPipeline:
    """Enhanced pipeline for diabetes prediction with SMOTE and feature engineering."""
    
    def __init__(self, data_path='data/clinical_data.csv', random_state=42):
        """Initialize the pipeline."""
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalers = {}
        self.label_encoders = {}
        self.models = {}
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for diabetes prediction."""
        print("=" * 70)
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("=" * 70)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Data loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        
        # Check class balance
        n_negative = (self.df['diabetes'] == 0).sum()
        n_positive = (self.df['diabetes'] == 1).sum()
        print(f"\nClass distribution (BEFORE SMOTE):")
        print(f"  Negative (diabetes=0): {n_negative} samples ({n_negative/len(self.df)*100:.1f}%)")
        print(f"  Positive (diabetes=1): {n_positive} samples ({n_positive/len(self.df)*100:.1f}%)")
        print(f"  Imbalance ratio: {n_negative/n_positive:.2f}:1")
        
        # Handle missing values
        print(f"\n[+] Handling missing values...")
        imputer = SimpleImputer(strategy='median')
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.df[numerical_cols] = imputer.fit_transform(self.df[numerical_cols])
        print(f"✓ Missing values imputed")
        
        return self
    
    def create_diabetes_features(self):
        """Create diabetes-specific engineered features."""
        print(f"\n[+] Creating diabetes-specific features...")
        
        # Extract features and target
        X = self.df.drop(['diabetes', 'cad'], axis=1).copy()
        y = self.df['diabetes'].copy()
        
        # 1. Glucose-BMI Interaction (important for diabetes)
        X['glucose_bmi_interaction'] = X['glucose'] * X['bmi']
        
        # 2. Glucose to HDL ratio (cardiovascular health marker)
        X['glucose_hdl_ratio'] = X['glucose'] / (X['hdl'] + 1e-5)
        
        # 3. Lipid profile score (triglycerides + cholesterol - HDL)
        X['lipid_score'] = X['triglycerides'] + X['cholesterol'] - X['hdl']
        
        # 4. Insulin resistance proxy (glucose * insulin)
        X['insulin_resistance'] = X['glucose'] * X['insulin']
        
        # 5. Weight category based on BMI
        X['bmi_category'] = pd.cut(X['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                     labels=[1, 2, 3, 4]).astype(int)
        
        # 6. Age groups
        X['age_group'] = pd.cut(X['age'], bins=[0, 40, 50, 60, 100], 
                                  labels=[1, 2, 3, 4]).astype(int)
        
        # 7. Blood pressure category
        X['bp_category'] = pd.cut(X['blood_pressure'], bins=[0, 120, 130, 200], 
                                    labels=[1, 2, 3]).astype(int)
        
        # 8. Cholesterol to HDL ratio
        X['cholesterol_hdl_ratio'] = X['cholesterol'] / (X['hdl'] + 1e-5)
        
        # 9. LDL to HDL ratio
        X['ldl_hdl_ratio'] = X['ldl'] / (X['hdl'] + 1e-5)
        
        # 10. Total cholesterol risk
        X['cholesterol_risk'] = X['cholesterol'] - X['hdl']
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        X['gender_encoded'] = le_gender.fit_transform(X['gender'])
        
        le_smoking = LabelEncoder()
        X['smoking_encoded'] = le_smoking.fit_transform(X['smoking'])
        
        le_fh = LabelEncoder()
        X['family_history_encoded'] = le_fh.fit_transform(X['family_history'])
        
        # Drop original categorical columns
        X = X.drop(['gender', 'smoking', 'family_history'], axis=1)
        
        print(f"✓ Created 10 engineered features")
        print(f"✓ Total features: {X.shape[1]}")
        print(f"\nNew features created:")
        print(f"  - glucose_bmi_interaction")
        print(f"  - glucose_hdl_ratio")
        print(f"  - lipid_score")
        print(f"  - insulin_resistance")
        print(f"  - bmi_category")
        print(f"  - age_group")
        print(f"  - bp_category")
        print(f"  - cholesterol_hdl_ratio")
        print(f"  - ldl_hdl_ratio")
        print(f"  - cholesterol_risk")
        
        return X, y
    
    def apply_smote(self, X, y):
        """Apply SMOTE oversampling."""
        print(f"\n[+] Applying SMOTE oversampling...")
        
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        n_negative_after = (y_resampled == 0).sum()
        n_positive_after = (y_resampled == 1).sum()
        
        print(f"✓ SMOTE applied successfully")
        print(f"\nClass distribution (AFTER SMOTE):")
        print(f"  Negative (diabetes=0): {n_negative_after} samples")
        print(f"  Positive (diabetes=1): {n_positive_after} samples")
        print(f"  Balance: 1:1 (perfectly balanced)")
        
        return X_resampled, y_resampled
    
    def split_and_scale_data(self, X, y):
        """Split data and scale features."""
        print(f"\n[+] Splitting data (80-20 stratified)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Training set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print(f"\n[+] Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"✓ Features scaled using StandardScaler")
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return self
    
    def train_random_forest(self):
        """Train Random Forest with hyperparameter tuning."""
        print(f"\n{'='*70}")
        print(f"TRAINING: Random Forest with RandomizedSearchCV")
        print(f"{'='*70}")
        
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            rf, param_dist, n_iter=20, cv=5, scoring='roc_auc', 
            n_jobs=-1, random_state=self.random_state, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_rf = random_search.best_estimator_
        train_auc = roc_auc_score(self.y_train, best_rf.predict_proba(self.X_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, best_rf.predict_proba(self.X_test)[:, 1])
        
        print(f"\n✓ Best parameters: {random_search.best_params_}")
        print(f"✓ Best CV score: {random_search.best_score_:.4f}")
        print(f"✓ Train AUC: {train_auc:.4f}")
        print(f"✓ Test AUC: {test_auc:.4f}")
        
        self.models['RandomForest'] = best_rf
        return self
    
    def train_xgboost(self):
        """Train XGBoost with hyperparameter tuning."""
        print(f"\n{'='*70}")
        print(f"TRAINING: XGBoost with RandomizedSearchCV")
        print(f"{'='*70}")
        
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 1, 5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1],
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state, use_label_encoder=False, 
            eval_metric='logloss', n_jobs=-1
        )
        
        random_search = RandomizedSearchCV(
            xgb_model, param_dist, n_iter=20, cv=5, scoring='roc_auc',
            n_jobs=-1, random_state=self.random_state, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_xgb = random_search.best_estimator_
        train_auc = roc_auc_score(self.y_train, best_xgb.predict_proba(self.X_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, best_xgb.predict_proba(self.X_test)[:, 1])
        
        print(f"\n✓ Best parameters: {random_search.best_params_}")
        print(f"✓ Best CV score: {random_search.best_score_:.4f}")
        print(f"✓ Train AUC: {train_auc:.4f}")
        print(f"✓ Test AUC: {test_auc:.4f}")
        
        self.models['XGBoost'] = best_xgb
        return self
    
    def train_lightgbm(self):
        """Train LightGBM with hyperparameter tuning."""
        print(f"\n{'='*70}")
        print(f"TRAINING: LightGBM with RandomizedSearchCV")
        print(f"{'='*70}")
        
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, 20],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 30, 40, 50],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1],
        }
        
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state, n_jobs=-1, verbose=-1
        )
        
        random_search = RandomizedSearchCV(
            lgb_model, param_dist, n_iter=20, cv=5, scoring='roc_auc',
            n_jobs=-1, random_state=self.random_state, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_lgb = random_search.best_estimator_
        train_auc = roc_auc_score(self.y_train, best_lgb.predict_proba(self.X_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, best_lgb.predict_proba(self.X_test)[:, 1])
        
        print(f"\n✓ Best parameters: {random_search.best_params_}")
        print(f"✓ Best CV score: {random_search.best_score_:.4f}")
        print(f"✓ Train AUC: {train_auc:.4f}")
        print(f"✓ Test AUC: {test_auc:.4f}")
        
        self.models['LightGBM'] = best_lgb
        return self
    
    def perform_cross_validation(self):
        """Perform 5-fold cross-validation for all models."""
        print(f"\n{'='*70}")
        print(f"STEP 2: 5-FOLD CROSS-VALIDATION")
        print(f"{'='*70}")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            print(f"\n[+] Cross-validating {model_name}...")
            
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, cv=cv, 
                scoring='roc_auc', n_jobs=-1
            )
            
            self.cv_results[model_name] = {
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
            }
            
            print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
            print(f"  Mean AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self
    
    def evaluate_models(self):
        """Evaluate all models on test set."""
        print(f"\n{'='*70}")
        print(f"STEP 3: MODEL EVALUATION ON TEST SET")
        print(f"{'='*70}")
        
        results_summary = []
        
        for model_name, model in self.models.items():
            print(f"\n{'─'*70}")
            print(f"EVALUATING: {model_name}")
            print(f"{'─'*70}")
            
            # Predictions
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = model.predict(self.X_test)
            
            # Metrics
            auc = roc_auc_score(self.y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            
            # Classification report
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            cm = confusion_matrix(self.y_test, y_pred)
            
            print(f"\n  ROC-AUC: {auc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"\n  Confusion Matrix:")
            print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
            
            results_summary.append({
                'Model': model_name,
                'ROC-AUC': auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'CV Mean AUC': self.cv_results[model_name]['mean'],
                'CV Std AUC': self.cv_results[model_name]['std'],
                'fpr': fpr,
                'tpr': tpr,
            })
        
        self.results_summary = pd.DataFrame(results_summary)
        
        # Find best model
        best_idx = self.results_summary['ROC-AUC'].idxmax()
        self.best_model_name = self.results_summary.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"ROC-AUC: {self.results_summary.loc[best_idx, 'ROC-AUC']:.4f}")
        print(f"{'='*70}")
        
        return self
    
    def generate_plots(self):
        """Generate evaluation plots."""
        print(f"\n{'='*70}")
        print(f"STEP 4: GENERATING EVALUATION PLOTS")
        print(f"{'='*70}")
        
        os.makedirs('results', exist_ok=True)
        
        # 1. ROC Curves Comparison
        plt.figure(figsize=(12, 8))
        for _, row in self.results_summary.iterrows():
            plt.plot(row['fpr'], row['tpr'], label=f"{row['Model']} (AUC={row['ROC-AUC']:.4f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Diabetes Prediction Models (ENHANCED)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/diabetes_roc_curves_enhanced.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: results/diabetes_roc_curves_enhanced.png")
        plt.close()
        
        # 2. Model Performance Comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison - Diabetes Prediction', fontsize=16, fontweight='bold')
        
        metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean AUC']
        models = self.results_summary['Model'].tolist()
        colors = ['#FF6B6B' if m == self.best_model_name else '#4ECDC4' for m in models]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(models, self.results_summary[metric], color=colors, alpha=0.8, edgecolor='black')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/diabetes_model_comparison_enhanced.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: results/diabetes_model_comparison_enhanced.png")
        plt.close()
        
        # 3. Cross-Validation Results
        plt.figure(figsize=(12, 6))
        cv_means = [self.cv_results[m]['mean'] for m in models]
        cv_stds = [self.cv_results[m]['std'] for m in models]
        colors = ['#FF6B6B' if m == self.best_model_name else '#95E1D3' for m in models]
        
        x_pos = np.arange(len(models))
        plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=10, color=colors, alpha=0.8, edgecolor='black')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('ROC-AUC', fontsize=12)
        plt.title('5-Fold Cross-Validation Results - Diabetes Prediction', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, models)
        plt.ylim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            plt.text(i, mean + std + 0.02, f'{mean:.4f}', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/diabetes_cv_results_enhanced.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: results/diabetes_cv_results_enhanced.png")
        plt.close()
        
        # 4. Feature Importance (Best Model)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features
            
            plt.figure(figsize=(10, 8))
            plt.barh(self.X_train.columns[indices], importances[indices], color='#38BEC9', edgecolor='black')
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Top 15 Feature Importances - {self.best_model_name}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('results/diabetes_feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: results/diabetes_feature_importance_enhanced.png")
            plt.close()
        
        print(f"\n✓ All plots generated successfully")
        return self
    
    def save_best_model(self):
        """Save the best model."""
        print(f"\n[+] Saving best model...")
        
        os.makedirs('models', exist_ok=True)
        model_path = f'models/best_diabetes_model_{self.best_model_name}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"✓ Model saved: {model_path}")
        
        # Also save metadata
        metadata = {
            'model_name': self.best_model_name,
            'test_auc': self.results_summary[self.results_summary['Model'] == self.best_model_name]['ROC-AUC'].values[0],
            'cv_mean_auc': self.results_summary[self.results_summary['Model'] == self.best_model_name]['CV Mean AUC'].values[0],
            'features': self.X_train.columns.tolist(),
        }
        
        import json
        metadata_path = 'models/diabetes_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved: {metadata_path}")
        return self
    
    def print_summary(self):
        """Print final summary."""
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        
        print(f"\n{'SUMMARY REPORT':^70}")
        print(f"{'─'*70}")
        print(f"\n1. MODELS TRAINED:")
        for model_name in self.models.keys():
            print(f"   ✓ {model_name}")
        
        print(f"\n2. PERFORMANCE METRICS:")
        print(self.results_summary[['Model', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
        
        print(f"\n3. CROSS-VALIDATION RESULTS:")
        for model_name, cv_result in self.cv_results.items():
            print(f"   {model_name}: {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")
        
        print(f"\n4. BEST MODEL: {self.best_model_name}")
        best_row = self.results_summary[self.results_summary['Model'] == self.best_model_name].iloc[0]
        print(f"   ROC-AUC: {best_row['ROC-AUC']:.4f}")
        print(f"   Accuracy: {best_row['Accuracy']:.4f}")
        print(f"   Precision: {best_row['Precision']:.4f}")
        print(f"   Recall: {best_row['Recall']:.4f}")
        print(f"   F1-Score: {best_row['F1-Score']:.4f}")
        print(f"   CV Mean AUC: {best_row['CV Mean AUC']:.4f} ± {best_row['CV Std AUC']:.4f}")
        
        goal_achieved = best_row['ROC-AUC'] > 0.85
        status = "✓ ACHIEVED" if goal_achieved else "✗ NEEDS IMPROVEMENT"
        print(f"\n5. TARGET: ROC-AUC > 0.85")
        print(f"   {status}: {best_row['ROC-AUC']:.4f}")
        
        print(f"\n6. OUTPUTS SAVED:")
        print(f"   ✓ Best model: models/best_diabetes_model_{self.best_model_name}.pkl")
        print(f"   ✓ Plots: results/diabetes_*_enhanced.png")
        print(f"   ✓ Metadata: models/diabetes_model_metadata.json")
        print(f"\n{'='*70}\n")
        
        return self
    
    def run(self):
        """Execute the complete pipeline."""
        (self
         .load_and_prepare_data()
        )
        
        # Create features and apply SMOTE
        X, y = self.create_diabetes_features()
        X, y = self.apply_smote(X, y)
        
        # Split and scale
        self.split_and_scale_data(X, y)
        
        # Train models
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        
        # Evaluate
        self.perform_cross_validation()
        self.evaluate_models()
        
        # Visualize and save
        self.generate_plots()
        self.save_best_model()
        self.print_summary()


if __name__ == "__main__":
    pipeline = DiabetesEnhancedPipeline()
    pipeline.run()

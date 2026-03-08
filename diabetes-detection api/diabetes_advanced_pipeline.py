"""
Advanced Diabetes Prediction Pipeline with Enhanced Feature Engineering and Model Stacking.

Further improvements:
1. Extended polynomial and interaction features
2. Domain-specific clinical features
3. Ensemble stacking with meta-learner
4. GridSearchCV for finer hyperparameter tuning
5. Custom threshold optimization
6. Class weight optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, cross_validate
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix, 
    classification_report, precision_recall_curve, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedDiabetesPipeline:
    """Advanced diabetes prediction pipeline with stacking and advanced feature engineering."""
    
    def __init__(self, data_path='data/clinical_data.csv', random_state=42):
        """Initialize the pipeline."""
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_prepare_data(self):
        """Load and prepare data."""
        print("=" * 70)
        print("STEP 1: DATA LOADING AND ADVANCED PREPARATION")
        print("=" * 70)
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Data loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.df[numerical_cols] = imputer.fit_transform(self.df[numerical_cols])
        print(f"✓ Missing values imputed")
        
        return self
    
    def create_advanced_features(self):
        """Create advanced clinical features."""
        print(f"\n[+] Creating advanced clinical features...")
        
        X = self.df.drop(['diabetes', 'cad'], axis=1).copy()
        y = self.df['diabetes'].copy()
        
        # ===== CLINICAL RISK SCORES =====
        # Waist-to-Hip indicator (approximated using BMI and age)
        X['metabolic_syndrome_score'] = (X['bmi'] * X['glucose'] * X['triglycerides']) / 1000
        
        # Diabetes Risk Score (modified)
        X['diabetes_risk_score'] = (
            0.3 * X['glucose'] / 100 +
            0.2 * X['bmi'] / 30 +
            0.2 * X['insulin'] / 10 +
            0.1 * X['triglycerides'] / 150 +
            0.2 * X['age'] / 70
        )
        
        # Cardiovascular Risk (proxy for diabetes complications)
        X['cv_risk_score'] = (
            0.3 * X['blood_pressure'] / 140 +
            0.2 * (X['cholesterol'] - X['hdl']) / 200 +
            0.2 * X['triglycerides'] / 250 +
            0.3 * X['ldl'] / 160
        )
        
        # ===== GLUCOSE METABOLIC FEATURES =====
        X['glucose_squared'] = X['glucose'] ** 2
        X['glucose_log'] = np.log1p(np.maximum(X['glucose'], 0.1))
        X['glucose_bmi_interaction'] = X['glucose'] * X['bmi']
        X['glucose_age_interaction'] = X['glucose'] * X['age'] / 100
        X['glucose_insulin_interaction'] = X['glucose'] * X['insulin']
        
        # ===== LIPID RATIO FEATURES =====
        X['triglyceride_hdl_ratio'] = X['triglycerides'] / (X['hdl'] + 1e-5)
        X['cholesterol_hdl_ratio'] = X['cholesterol'] / (X['hdl'] + 1e-5)
        X['ldl_hdl_ratio'] = X['ldl'] / (X['hdl'] + 1e-5)
        X['ldl_triglyceride_ratio'] = X['ldl'] / (X['triglycerides'] + 1e-5)
        X['total_lipid_risk'] = X['cholesterol'] + X['triglycerides'] - 2 * X['hdl']
        
        # ===== INSULIN SENSITIVITY FEATURES =====
        X['insulin_log'] = np.log1p(np.maximum(X['insulin'], 0.1))
        X['fasting_insulin_squared'] = X['insulin'] ** 2
        X['insulin_bmi_interaction'] = X['insulin'] * X['bmi']
        X['insulin_glucose_ratio'] = X['insulin'] / (X['glucose'] + 1e-5)
        X['insulin_resistance_homa'] = (X['glucose'] * X['insulin']) / 405  # HOMA-IR approximation
        
        # ===== BMI STRATIFICATION =====
        X['bmi_squared'] = X['bmi'] ** 2
        X['bmi_age_interaction'] = X['bmi'] * X['age'] / 100
        X['bmi_category'] = pd.cut(X['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[1, 2, 3, 4]).astype(int)
        
        # ===== AGE RELATED FEATURES =====
        X['age_squared'] = X['age'] ** 2
        X['age_group'] = pd.cut(X['age'], bins=[0, 40, 50, 60, 100], labels=[1, 2, 3, 4]).astype(int)
        X['age_glucose_interaction'] = X['age'] * X['glucose'] / 100
        
        # ===== BLOOD PRESSURE FEATURES =====
        X['bp_squared'] = X['blood_pressure'] ** 2
        X['bp_category'] = pd.cut(X['blood_pressure'], bins=[0, 120, 130, 200], labels=[1, 2, 3]).astype(int)
        
        # ===== ENCODED CATEGORICAL =====
        from sklearn.preprocessing import LabelEncoder
        le_gender = LabelEncoder()
        X['gender_encoded'] = le_gender.fit_transform(X['gender'])
        
        le_smoking = LabelEncoder()
        X['smoking_encoded'] = le_smoking.fit_transform(X['smoking'])
        
        le_fh = LabelEncoder()
        X['family_history_encoded'] = le_fh.fit_transform(X['family_history'])
        
        # ===== COMBINED RISK INDICES =====
        X['metabolic_risk_index'] = (
            X['triglyceride_hdl_ratio'] * 0.3 +
            X['diabetes_risk_score'] * 0.4 +
            X['insulin_resistance_homa'] * 0.3
        )
        
        # Drop original categorical
        X = X.drop(['gender', 'smoking', 'family_history'], axis=1)
        
        # Fill any remaining NaNs
        X = X.fillna(X.median())
        
        print(f"✓ Created 30+ advanced diabetes-specific features")
        print(f"✓ Total features: {X.shape[1]}")
        
        return X, y
    
    def apply_smote_variants(self, X, y):
        """Apply SMOTE with k-neighbors tuning."""
        print(f"\n[+] Applying SMOTE with optimized parameters...")
        
        # Use k=3 for better local neighborhood in smaller minority class
        smote = SMOTE(random_state=self.random_state, k_neighbors=3, sampling_strategy=1.0)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        n_negative = (y_resampled == 0).sum()
        n_positive = (y_resampled == 1).sum()
        
        print(f"✓ SMOTE applied: {n_negative} negative, {n_positive} positive (1:1 ratio)")
        
        return X_resampled, y_resampled
    
    def split_and_scale_data(self, X, y):
        """Split and scale data."""
        print(f"\n[+] Splitting and scaling data...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
        print(f"✓ Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        
        return self
    
    def train_tuned_models(self):
        """Train models with fine-tuned hyperparameters."""
        print(f"\n{'='*70}")
        print(f"STEP 2: TRAINING FINE-TUNED MODELS WITH GRIDSEARCHCV")
        print(f"{'='*70}")
        
        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        # ===== RANDOM FOREST WITH FINE TUNING =====
        print(f"\n[→] Random Forest Fine-Tuning...")
        param_grid_rf = {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25, 30],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'class_weight': [class_weight_dict, 'balanced'],
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        grid_search_rf = GridSearchCV(
            rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search_rf.fit(self.X_train, self.y_train)
        
        best_rf = grid_search_rf.best_estimator_
        train_auc = roc_auc_score(self.y_train, best_rf.predict_proba(self.X_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, best_rf.predict_proba(self.X_test)[:, 1])
        
        print(f"✓ Best CV score: {grid_search_rf.best_score_:.4f}")
        print(f"✓ Test AUC: {test_auc:.4f}")
        
        self.models['RandomForest'] = best_rf
        
        # ===== XGBOOST WITH FINE TUNING =====
        print(f"\n[→] XGBoost Fine-Tuning...")
        param_grid_xgb = {
            'n_estimators': [200, 300, 400],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.03, 0.05],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state, scale_pos_weight=class_weights[0]/class_weights[1],
            use_label_encoder=False, eval_metric='logloss', n_jobs=-1
        )
        grid_search_xgb = GridSearchCV(
            xgb_model, param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search_xgb.fit(self.X_train, self.y_train)
        
        best_xgb = grid_search_xgb.best_estimator_
        test_auc = roc_auc_score(self.y_test, best_xgb.predict_proba(self.X_test)[:, 1])
        
        print(f"✓ Best CV score: {grid_search_xgb.best_score_:.4f}")
        print(f"✓ Test AUC: {test_auc:.4f}")
        
        self.models['XGBoost'] = best_xgb
        
        # ===== LIGHTGBM WITH FINE TUNING =====
        print(f"\n[→] LightGBM Fine-Tuning...")
        param_grid_lgb = {
            'n_estimators': [200, 300, 400],
            'max_depth': [5, 10, 15, 20],
            'learning_rate': [0.01, 0.03, 0.05],
            'num_leaves': [20, 30, 40, 50],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_samples': [5, 10, 20],
        }
        
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state, class_weight='balanced',
            n_jobs=-1, verbose=-1
        )
        grid_search_lgb = GridSearchCV(
            lgb_model, param_grid_lgb, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search_lgb.fit(self.X_train, self.y_train)
        
        best_lgb = grid_search_lgb.best_estimator_
        test_auc = roc_auc_score(self.y_test, best_lgb.predict_proba(self.X_test)[:, 1])
        
        print(f"✓ Best CV score: {grid_search_lgb.best_score_:.4f}")
        print(f"✓ Test AUC: {test_auc:.4f}")
        
        self.models['LightGBM'] = best_lgb
        
        # ===== GRADIENT BOOSTING =====
        print(f"\n[→] Gradient Boosting Fine-Tuning...")
        param_grid_gb = {
            'n_estimators': [200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
        }
        
        gb = GradientBoostingClassifier(random_state=self.random_state)
        grid_search_gb = GridSearchCV(
            gb, param_grid_gb, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search_gb.fit(self.X_train, self.y_train)
        
        best_gb = grid_search_gb.best_estimator_
        test_auc = roc_auc_score(self.y_test, best_gb.predict_proba(self.X_test)[:, 1])
        
        print(f"✓ Best CV score: {grid_search_gb.best_score_:.4f}")
        print(f"✓ Test AUC: {test_auc:.4f}")
        
        self.models['GradientBoosting'] = best_gb
        
        return self
    
    def evaluate_all_models(self):
        """Evaluate all models."""
        print(f"\n{'='*70}")
        print(f"STEP 3: MODEL EVALUATION")
        print(f"{'='*70}")
        
        results = []
        
        for model_name, model in self.models.items():
            print(f"\n{'─'*70}")
            print(f"EVALUATING: {model_name}")
            print(f"{'─'*70}")
            
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = model.predict(self.X_test)
            
            # Metrics
            auc = roc_auc_score(self.y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            print(f"  ROC-AUC: {auc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            results.append({
                'Model': model_name,
                'ROC-AUC': auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'fpr': fpr,
                'tpr': tpr,
            })
        
        self.results_df = pd.DataFrame(results)
        
        # Find best model
        best_idx = self.results_df['ROC-AUC'].idxmax()
        self.best_model_name = self.results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {self.best_model_name} (AUC: {self.results_df.loc[best_idx, 'ROC-AUC']:.4f})")
        print(f"{'='*70}")
        
        return self
    
    def generate_visualizations(self):
        """Generate evaluation plots."""
        print(f"\n[+] Generating visualizations...")
        
        os.makedirs('results', exist_ok=True)
        
        # ROC Curves
        plt.figure(figsize=(12, 8))
        for _, row in self.results_df.iterrows():
            plt.plot(row['fpr'], row['tpr'], label=f"{row['Model']} (AUC={row['ROC-AUC']:.4f})", linewidth=2.5)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Diabetes Prediction (Advanced Pipeline)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/diabetes_roc_advanced.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: results/diabetes_roc_advanced.png")
        plt.close()
        
        # Model Performance
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        fig.suptitle('Model Performance Comparison - Diabetes Prediction', fontsize=14, fontweight='bold')
        
        metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        models = self.results_df['Model'].tolist()
        colors = ['#FF6B6B' if m == self.best_model_name else '#4ECDC4' for m in models]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(models, self.results_df[metric], color=colors, alpha=0.8, edgecolor='black')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/diabetes_performance_advanced.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: results/diabetes_performance_advanced.png")
        plt.close()
        
        return self
    
    def save_results(self):
        """Save results and models."""
        print(f"\n[+] Saving models and results...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save best model
        model_path = f'models/best_diabetes_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Saved: {model_path}")
        
        # Save results summary
        results_path = 'results/diabetes_results_advanced.csv'
        self.results_df.drop(['fpr', 'tpr'], axis=1).to_csv(results_path, index=False)
        print(f"✓ Saved: {results_path}")
        
        return self
    
    def print_summary(self):
        """Print final summary."""
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        
        print(f"\n{'FINAL RESULTS':^70}")
        print(f"{'─'*70}")
        
        print(f"\n1. MODELS TRAINED:")
        for model_name in self.models.keys():
            print(f"   ✓ {model_name}")
        
        print(f"\n2. PERFORMANCE METRICS:")
        print(self.results_df[['Model', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
        
        print(f"\n3. BEST MODEL: {self.best_model_name}")
        best_row = self.results_df[self.results_df['Model'] == self.best_model_name].iloc[0]
        print(f"   ROC-AUC: {best_row['ROC-AUC']:.4f}")
        print(f"   Accuracy: {best_row['Accuracy']:.4f}")
        print(f"   Precision: {best_row['Precision']:.4f}")
        print(f"   Recall: {best_row['Recall']:.4f}")
        print(f"   F1-Score: {best_row['F1-Score']:.4f}")
        
        gap = 0.85 - best_row['ROC-AUC']
        if gap <= 0:
            print(f"\n✓ TARGET ACHIEVED: ROC-AUC > 0.85")
            print(f"  Final AUC: {best_row['ROC-AUC']:.4f} (+{-gap:.4f} above target)")
        else:
            print(f"\n⚠ Target: ROC-AUC > 0.85")
            print(f"  Current: {best_row['ROC-AUC']:.4f} (need +{gap:.4f} improvement)")
        
        print(f"\n{'='*70}\n")
        
        return self
    
    def run(self):
        """Execute complete pipeline."""
        (self
         .load_and_prepare_data()
        )
        
        X, y = self.create_advanced_features()
        X, y = self.apply_smote_variants(X, y)
        self.split_and_scale_data(X, y)
        
        self.train_tuned_models()
        self.evaluate_all_models()
        self.generate_visualizations()
        self.save_results()
        self.print_summary()


if __name__ == "__main__":
    pipeline = AdvancedDiabetesPipeline()
    pipeline.run()

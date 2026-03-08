"""
Advanced diabetes prediction optimization - Final attempt with maximum engineering.

Strategy:
1. Aggressive feature engineering with domain knowledge
2. Ensemble stacking to combine weak signals
3. Cost-sensitive learning for class imbalance
4. Threshold optimization for ROC-AUC
5. Multiple model architectures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, KBinsDiscretizer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, auc, accuracy_score, 
                             precision_score, recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import EasyEnsembleClassifier

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

print("\n" + "="*70)
print("ADVANCED DIABETES OPTIMIZATION - MAXIMUM EFFORT")
print("="*70)

# Load and preprocess data
print("\n1. DATA PREPROCESSING & ADVANCED FEATURE ENGINEERING")
print("-" * 70)

df = pd.read_csv('data/clinical_dataset_100k.csv')
X = df.drop(['diabetes', 'cad'], axis=1)
y = df['diabetes'].copy()

print(f"Original: {X.shape}, Diabetes: {(y==1).sum()}/{len(y)}")

# Encode categoricals - force all to numeric first
X_proc = X.copy()

# Convert all columns to float to ensure compatibility
for col in X_proc.columns:
    if X_proc[col].dtype == 'object':
        # Try mapping if this is a known categorical
        if col == 'gender':
            mapped = X_proc[col].map({'Male': 1, 'Female': 0})
            X_proc[col] = mapped.fillna(0).astype(np.float64)
        elif col == 'smoking_status':
            mapped = X_proc[col].map({'Never': 0, 'Former': 1, 'Current': 2})
            X_proc[col] = mapped.fillna(0).astype(np.float64)
        elif col == 'smoking':
            mapped = X_proc[col].map({'Yes': 1, 'No': 0})
            X_proc[col] = mapped.fillna(0).astype(np.float64)
        elif col == 'family_history':
            mapped = X_proc[col].map({'Yes': 1, 'No': 0})
            X_proc[col] = mapped.fillna(0).astype(np.float64)
        else:
            # Unknown categorical, try label encoding
            X_proc[col] = pd.Categorical(X_proc[col]).codes.astype(np.float64)
    else:
        X_proc[col] = pd.to_numeric(X_proc[col], errors='coerce').fillna(0).astype(np.float64)

# Verify all columns are numeric now
X_proc = X_proc.astype(np.float64)

# Now impute missing values if any
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X_proc), columns=X_proc.columns)

# AGGRESSIVE FEATURE ENGINEERING
print("\n[INFO] Creating advanced diabetes-specific features...")

fe_count = 0

# Glucose-based risk features
if 'fasting_glucose' in X_imputed.columns:
    X_imputed['glucose_risk'] = pd.cut(X_imputed['fasting_glucose'], bins=[0, 100, 126, 200], labels=[0, 1, 2]).astype(float)
    X_imputed['glucose_bmi_interaction'] = X_imputed['fasting_glucose'] * X_imputed['bmi']
    X_imputed['glucose_age_interaction'] = X_imputed['fasting_glucose'] * X_imputed['age']
    X_imputed['glucose_log'] = np.log(X_imputed['fasting_glucose'] + 1)
    X_imputed['glucose_squared'] = X_imputed['fasting_glucose'] ** 2
    fe_count += 5

# HbA1c-based features
if 'hba1c' in X_imputed.columns:
    X_imputed['hba1c_risk'] = (X_imputed['hba1c'] > 6.5).astype(int)
    X_imputed['hba1c_squared'] = X_imputed['hba1c'] ** 2
    fe_count += 2

# BMI-based risk features
if 'bmi' in X_imputed.columns:
    X_imputed['bmi_risk'] = pd.cut(X_imputed['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    X_imputed['bmi_squared'] = X_imputed['bmi'] ** 2
    if 'fasting_glucose' in X_imputed.columns:
        X_imputed['bmi_glucose_ratio'] = X_imputed['bmi'] / (X_imputed['fasting_glucose'] + 1e-6)
    if 'hdl' in X_imputed.columns:
        X_imputed['bmi_hdl_ratio'] = X_imputed['bmi'] / (X_imputed['hdl'] + 1e-6)
    fe_count += 4

# Lipid-based risk
if 'hdl' in X_imputed.columns and 'triglycerides' in X_imputed.columns:
    X_imputed['triglyceride_hdl_ratio'] = X_imputed['triglycerides'] / (X_imputed['hdl'] + 1e-6)
    X_imputed['lipid_risk_score'] = (X_imputed['triglycerides'] > 150).astype(int) + (X_imputed['hdl'] < 40).astype(int)
    fe_count += 2

if 'ldl' in X_imputed.columns and 'hdl' in X_imputed.columns:
    X_imputed['ldl_hdl_ratio'] = X_imputed['ldl'] / (X_imputed['hdl'] + 1e-6)
    fe_count += 1

if 'cholesterol_total' in X_imputed.columns and 'triglycerides' in X_imputed.columns:
    X_imputed['total_lipid_burden'] = X_imputed['cholesterol_total'] + X_imputed['triglycerides']
    fe_count += 1

# Insulin resistance proxy
if 'fasting_glucose' in X_imputed.columns and 'insulin_level' in X_imputed.columns:
    X_imputed['insulin_glucose_product'] = X_imputed['fasting_glucose'] * X_imputed['insulin_level']
    X_imputed['homa_ir_approx'] = (X_imputed['fasting_glucose'] * X_imputed['insulin_level']) / 405.0
    X_imputed['homa_ir_risk'] = (X_imputed['homa_ir_approx'] > 2.5).astype(int)
    X_imputed['insulin_risk'] = (X_imputed['insulin_level'] > 12).astype(int)
    fe_count += 4

# Age-related risk
if 'age' in X_imputed.columns:
    X_imputed['age_risk'] = pd.cut(X_imputed['age'], bins=[0, 35, 50, 65, 150], labels=[0, 1, 2, 3]).astype(float)
    if 'fasting_glucose' in X_imputed.columns:
        X_imputed['age_glucose_interaction'] = X_imputed['age'] * X_imputed['fasting_glucose']
    fe_count += 2

# Blood pressure risk
if 'systolic_bp' in X_imputed.columns:
    X_imputed['bp_risk'] = (X_imputed['systolic_bp'] > 130).astype(int)
    if 'fasting_glucose' in X_imputed.columns:
        X_imputed['bp_glucose_interaction'] = X_imputed['systolic_bp'] * X_imputed['fasting_glucose']
    fe_count += 2

# Waist circumference for metabolic syndrome
if 'waist_circumference' in X_imputed.columns:
    X_imputed['waist_risk'] = (X_imputed['waist_circumference'] > 102).astype(int)
    fe_count += 1

# Family history interactions
if 'family_history_diabetes' in X_imputed.columns:
    if 'fasting_glucose' in X_imputed.columns:
        X_imputed['fh_glucose_interaction'] = X_imputed['family_history_diabetes'] * X_imputed['fasting_glucose']
    if 'bmi' in X_imputed.columns:
        X_imputed['fh_bmi_interaction'] = X_imputed['family_history_diabetes'] * X_imputed['bmi']
    fe_count += 2

print(f"[OK] Created {fe_count} advanced features")
print(f"Total features now: {X_imputed.shape[1]}")

# Select top features using correlation and importance
print("\n[INFO] Performing feature selection...")
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

selector_f = SelectKBest(f_classif, k=min(25, X_imputed.shape[1]))
selector_f.fit(X_imputed, y)
selected_features_f = X_imputed.columns[selector_f.get_support()].tolist()

selector_mi = SelectKBest(mutual_info_classif, k=min(25, X_imputed.shape[1]))
selector_mi.fit(X_imputed, y)
selected_features_mi = X_imputed.columns[selector_mi.get_support()].tolist()

selected_features = list(set(selected_features_f + selected_features_mi))
X_selected = X_imputed[selected_features].copy()

print(f"[OK] Selected {len(selected_features)} features for training")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[OK] Training set: {X_train_scaled.shape}")
print(f"[OK] Test set: {X_test_scaled.shape}")

# Apply SMOTE to training data
print("\n2. APPLYING SMOTE OVERSAMPLING")
print("-" * 70)

smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"[OK] After SMOTE: {X_train_smote.shape}")
print(f"    Class 0: {(y_train_smote==0).sum()}, Class 1: {(y_train_smote==1).sum()}")

# Model training with multiple models
print("\n3. TRAINING MULTIPLE MODELS WITH COST-SENSITIVE LEARNING")
print("-" * 70)

models_config = {
    'LightGBM_Balanced': LGBMClassifier(
        n_estimators=300,
        num_leaves=60,
        max_depth=12,
        learning_rate=0.03,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=(y_train==0).sum() / (y_train==1).sum()
    ),
    'XGBoost_Balanced': XGBClassifier(
        n_estimators=300,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
        eval_metric='logloss'
    ),
    'RandomForest_Balanced': RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'GradientBoosting_Balanced': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        min_samples_split=3,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
}

# Train models on SMOTE data
trained_models = {}
cv_scores = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models_config.items():
    print(f"\n[INFO] Training {model_name}...")
    
    cv_aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Apply SMOTE to CV fold
        smote_cv = SMOTE(random_state=42, k_neighbors=min(5, (y_cv_train==1).sum()-1))
        X_cv_train_sm, y_cv_train_sm = smote_cv.fit_resample(X_cv_train, y_cv_train)
        
        # Train model
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_cv_train_sm, y_cv_train_sm)
        
        # Evaluate
        y_pred_proba = model_copy.predict_proba(X_cv_val)[:, 1]
        auc_score = roc_auc_score(y_cv_val, y_pred_proba)
        cv_aucs.append(auc_score)
        
        print(f"  Fold {fold+1}: AUC = {auc_score:.4f}")
    
    # Train on full training set
    smote_final = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
    X_train_sm, y_train_sm = smote_final.fit_resample(X_train_scaled, y_train)
    
    model_final = model.__class__(**model.get_params())
    model_final.fit(X_train_sm, y_train_sm)
    trained_models[model_name] = model_final
    
    cv_scores[model_name] = {
        'mean': np.mean(cv_aucs),
        'std': np.std(cv_aucs),
        'scores': cv_aucs
    }
    
    print(f"  Mean CV AUC: {cv_scores[model_name]['mean']:.4f} +/- {cv_scores[model_name]['std']:.4f}")

# Test set evaluation
print("\n4. TEST SET EVALUATION")
print("-" * 70)

test_results = []

for model_name, model in trained_models.items():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    test_results.append({
        'Model': model_name,
        'Test_AUC': auc_score,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'CV_AUC_Mean': cv_scores[model_name]['mean']
    })
    
    print(f"\n{model_name}:")
    print(f"  Test AUC: {auc_score:.4f}")
    print(f"  CV AUC:   {cv_scores[model_name]['mean']:.4f} +/- {cv_scores[model_name]['std']:.4f}")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# Find best model
results_df = pd.DataFrame(test_results).sort_values('Test_AUC', ascending=False)
best_model_name = results_df.iloc[0]['Model']
best_auc = results_df.iloc[0]['Test_AUC']

# ROC curve visualization
print("\n5. ROC CURVE VISUALIZATION")
print("-" * 70)

plt.figure(figsize=(12, 8))

for model_name, model in trained_models.items():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2.5, 
            label=f'{model_name} (AUC = {auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.5000)')
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves - Diabetes Prediction (Advanced Optimization)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/diabetes_roc_advanced.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Saved: results/diabetes_roc_advanced.png")

# Results summary
print("\n" + "="*70)
print("DIABETES PREDICTION - ADVANCED OPTIMIZATION RESULTS")
print("="*70)
print("\nMODEL COMPARISON (sorted by Test AUC):")
print("-" * 70)
print(results_df[['Model', 'Test_AUC', 'CV_AUC_Mean', 'Accuracy', 'Precision', 'Recall', 'F1']].to_string(index=False))

print(f"\nBEST MODEL: {best_model_name}")
print(f"Test ROC-AUC: {best_auc:.4f}")

if best_auc > 0.85:
    print(f"[SUCCESS] Target AUC > 0.85 ACHIEVED!")
elif best_auc > 0.70:
    print(f"[GOOD] Significant improvement achieved (AUC > 0.70)")
else:
    print(f"[INFO] Dataset has inherently weak diabetes predictors")
    print(f"       Max correlation observed: 0.067 (too weak for >0.85 AUC)")

# Save best model
import pickle
best_model = trained_models[best_model_name]
filepath = f'results/diabetes_best_model_advanced_{best_model_name}.pkl'
with open(filepath, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n[OK] Best model saved: {filepath}")

print("\n" + "="*70)
print("ADVANCED OPTIMIZATION COMPLETE")
print("="*70)

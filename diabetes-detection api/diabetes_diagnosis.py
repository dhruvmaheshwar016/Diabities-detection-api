"""
Diagnostic script to understand why diabetes prediction is difficult.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.stats import pointbiserialr
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)

def analyze_dataset():
    """Analyze dataset characteristics."""
    print("\n" + "="*70)
    print("DIABETES PREDICTION - DATASET DIAGNOSIS")
    print("="*70)
    
    # Load data
    df = pd.read_csv('data/clinical_data.csv')
    
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   Shape: {df.shape}")
    print(f"   Diabetes distribution: {df['diabetes'].value_counts().to_dict()}")
    print(f"   CAD distribution: {df['cad'].value_counts().to_dict()}")
    
    # Get feature columns
    X = df.drop(['diabetes', 'cad'], axis=1)
    y_diabetes = df['diabetes']
    y_cad = df['cad']
    
    print(f"\n2. FEATURE ANALYSIS:")
    print(f"   Total features: {X.shape[1]}")
    print(f"   Numeric features: {X.select_dtypes(include=[np.number]).shape[1]}")
    print(f"   Categorical features: {X.select_dtypes(include=['object']).shape[1]}")
    
    # Compute correlation between features and diabetes
    print(f"\n3. FEATURE-DIABETES CORRELATION (Point-Biserial):")
    print("   " + "-"*60)
    
    numeric_features = X.select_dtypes(include=[np.number]).columns
    correlations = []
    
    for feat in numeric_features:
        if X[feat].notna().sum() > 0:
            corr, pval = pointbiserialr(y_diabetes, X[feat].fillna(X[feat].mean()))
            correlations.append({
                'Feature': feat,
                'Correlation': abs(corr),
                'P-value': pval
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    print(corr_df.to_string(index=False))
    
    # Compare with CAD
    print(f"\n4. FEATURE-CAD CORRELATION (Point-Biserial):")
    print("   " + "-"*60)
    
    correlations_cad = []
    for feat in numeric_features:
        if X[feat].notna().sum() > 0:
            corr, pval = pointbiserialr(y_cad, X[feat].fillna(X[feat].mean()))
            correlations_cad.append({
                'Feature': feat,
                'Correlation': abs(corr),
                'P-value': pval
            })
    
    corr_cad_df = pd.DataFrame(correlations_cad).sort_values('Correlation', ascending=False)
    print(corr_cad_df.to_string(index=False))
    
    # Model predictability test
    print(f"\n5. BASELINE MODEL CROSS-VALIDATION AUC:")
    print("   " + "-"*60)
    
    # Preprocess
    X_prep = X.copy()
    X_prep = X_prep.fillna(X_prep.mean(numeric_only=True))
    
    # Encode categoricals
    X_prep['gender'] = X_prep['gender'].map({'Male': 1, 'Female': 0})
    X_prep['smoking'] = X_prep['smoking'].map({'Yes': 1, 'No': 0})
    X_prep['family_history'] = X_prep['family_history'].map({'Yes': 1, 'No': 0})
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_prep)
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'LightGBM': None
    }
    
    for model_name, model in models.items():
        if model_name == 'LightGBM':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        
        scores_diabetes = cross_val_score(model, X_scaled, y_diabetes, cv=5, scoring='roc_auc')
        scores_cad = cross_val_score(model, X_scaled, y_cad, cv=5, scoring='roc_auc')
        
        print(f"   {model_name}:")
        print(f"     Diabetes AUC:  {scores_diabetes.mean():.4f} +/- {scores_diabetes.std():.4f}")
        print(f"     CAD AUC:       {scores_cad.mean():.4f} +/- {scores_cad.std():.4f}")
    
    # Feature importance analysis
    print(f"\n6. FEATURE IMPORTANCE (RandomForest):")
    print("   " + "-"*60)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y_diabetes)
    
    importances = pd.DataFrame({
        'Feature': X_prep.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importances.head(12).to_string(index=False))
    
    # Issue diagnosis
    print(f"\n7. DIAGNOSIS & INSIGHTS:")
    print("   " + "-"*60)
    
    max_corr_diabetes = corr_df['Correlation'].max()
    max_corr_cad = corr_cad_df['Correlation'].max()
    
    print(f"   - Max feature-diabetes correlation: {max_corr_diabetes:.4f}")
    print(f"   - Max feature-CAD correlation: {max_corr_cad:.4f}")
    print(f"   - Correlation ratio (CAD/Diabetes): {max_corr_cad/max_corr_diabetes:.2f}x")
    print()
    
    if max_corr_diabetes < 0.15:
        print("   FINDING: Diabetes features are weakly correlated with outcome")
        print("   - This explains why baseline AUC ~0.50 (random performance)")
        print("   - The dataset may lack strong diabetes predictors")
        print("   - Consider: feature engineering, data quality checks, or different approach")
    
    if max_corr_cad > 0.3:
        print(f"\n   CONCLUSION: CAD is much more predictable ({max_corr_cad:.3f}) than Diabetes ({max_corr_diabetes:.3f})")
        print("   - This is why CAD achieves AUC=1.0 while Diabetes struggles")
    
    return df, X_scaled, y_diabetes, y_cad, X_prep

if __name__ == '__main__':
    analyze_dataset()

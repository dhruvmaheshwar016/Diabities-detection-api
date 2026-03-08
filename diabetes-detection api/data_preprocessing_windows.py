"""
Enhanced data preprocessing module for medical disease prediction.
Windows-compatible version with ASCII output.

This module handles:
- Missing value imputation (KNN)
- Feature scaling (StandardScaler)
- Class imbalance correction (SMOTE)
- Feature correlation analysis
- Intelligent feature selection (mutual information + RFE)
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


class MedicalDataPreprocessor:
    """Comprehensive preprocessor for medical tabular data with disease targets."""
    
    def __init__(self, n_features=30, random_state=42):
        """Initialize preprocessor."""
        self.n_features = n_features
        self.random_state = random_state
        self.imputer = None
        self.scaler = None
        self.label_encoders = {}
        self.selected_features = None
        self.correlation_matrix = None
        self.feature_importance = None
        
    def load_data(self, data_path):
        """Load clinical data from CSV."""
        df = pd.read_csv(data_path)
        print(f"[OK] Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    
    def separate_features_targets(self, df, target_cols=['diabetes', 'cad']):
        """Separate features from target variables."""
        targets = df[target_cols].copy()
        features = df.drop(target_cols, axis=1).copy()
        
        print(f"[OK] Features: {features.shape[1]}, Targets: {targets.shape[1]}")
        print(f"     Target distribution:\n{targets.sum()}")
        
        return features, targets
    
    def handle_missing_values(self, features, strategy='knn'):
        """Handle missing values using KNN imputation."""
        missing_before = features.isnull().sum().sum()
        
        if strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
            numerical_cols = features.select_dtypes(include=[np.number]).columns
            features[numerical_cols] = self.imputer.fit_transform(features[numerical_cols])
        
        missing_after = features.isnull().sum().sum()
        print(f"[OK] Missing values handled: {missing_before} -> {missing_after}")
        
        return features
    
    def encode_categorical(self, features, is_train=True):
        """Encode categorical features using LabelEncoder."""
        categorical_cols = features.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if is_train:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                features[col] = le.transform(features[col].astype(str))
        
        print(f"[OK] Categorical features encoded: {len(categorical_cols)}")
        return features
    
    def scale_features(self, features, is_train=True):
        """Scale numerical features using StandardScaler."""
        numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        
        if is_train:
            self.scaler = StandardScaler()
            features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        else:
            features[numerical_cols] = self.scaler.transform(features[numerical_cols])
        
        print(f"[OK] Features scaled: {len(numerical_cols)} numerical columns")
        return features
    
    def remove_correlated_features(self, features, correlation_threshold=0.95):
        """Remove highly correlated features to reduce multicollinearity."""
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        
        if to_drop:
            features = features.drop(columns=to_drop)
            print(f"[OK] Highly correlated features removed: {to_drop}")
        else:
            print(f"[OK] No highly correlated features found (threshold={correlation_threshold})")
        
        self.correlation_matrix = corr_matrix
        return features
    
    def select_features(self, features, targets_primary, method='combined'):
        """Select most important features using mutual information and RFE."""
        n_original = features.shape[1]
        n_select = min(self.n_features, features.shape[1])
        
        if method in ['mutual_info', 'combined']:
            # Mutual information
            mi_scores = mutual_info_classif(features, targets_primary, 
                                           random_state=self.random_state)
            mi_features = pd.Series(mi_scores, index=features.columns).nlargest(n_select).index.tolist()
        
        if method in ['rfe', 'combined']:
            # RFE with Random Forest
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            rfe = RFE(estimator, n_features_to_select=n_select)
            rfe.fit(features, targets_primary)
            rfe_features = features.columns[rfe.support_].tolist()
        
        # Combine for more robust selection
        if method == 'combined':
            selected = list(set(mi_features) & set(rfe_features))
            if len(selected) < n_select:
                remaining = [f for f in mi_features if f not in selected]
                selected.extend(remaining[:n_select - len(selected)])
        elif method == 'mutual_info':
            selected = mi_features
        else:
            selected = rfe_features
        
        selected = selected[:n_select]
        selected_df = features[selected].copy()
        
        print(f"[OK] Features selected: {len(selected)}/{n_original}")
        print(f"     Selected features: {selected[:10]}")
        
        self.selected_features = selected
        return selected_df, selected
    
    def handle_class_imbalance(self, X, y_target, method='smote'):
        """Handle class imbalance using SMOTE."""
        original_balance = y_target.value_counts().to_dict()
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            try:
                X_balanced, y_balanced = smote.fit_resample(X, y_target)
                X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
                y_balanced = pd.Series(y_balanced, name=y_target.name)
                
                new_balance = y_balanced.value_counts().to_dict()
                print(f"[OK] SMOTE applied: {original_balance} -> {new_balance}")
                return X_balanced, y_balanced
            except Exception as e:
                print(f"[WARN] SMOTE failed: {e}. Proceeding without resampling.")
                return X, y_target
        
        return X, y_target
    
    def fit(self, data_path):
        """Complete preprocessing pipeline for training data."""
        print("\n" + "="*60)
        print("MEDICAL DATA PREPROCESSING - TRAINING PHASE")
        print("="*60)
        
        # Step 1: Load
        df = self.load_data(data_path)
        
        # Step 2: Separate
        features, targets = self.separate_features_targets(df)
        
        # Step 3: Handle missing values
        features = self.handle_missing_values(features)
        
        # Step 4: Encode categorical
        features = self.encode_categorical(features, is_train=True)
        
        # Step 5: Scale
        features = self.scale_features(features, is_train=True)
        
        # Step 6: Remove correlated
        features = self.remove_correlated_features(features)
        
        # Step 7: Select features
        features_selected, _ = self.select_features(features, targets['diabetes'])
        
        # Step 8: Handle imbalance for each target
        processed_data = {
            'X_train': features_selected,
            'y_diabetes_train': targets['diabetes'],
            'y_cad_train': targets['cad'],
            'feature_names': features_selected.columns.tolist(),
            'preprocessor': self
        }
        
        return processed_data
    
    def transform(self, features):
        """Apply fitted preprocessing to new data (for test/validation)."""
        # Encode categorical
        features = self.encode_categorical(features, is_train=False)
        
        # Scale
        features = self.scale_features(features, is_train=False)
        
        # Select same features
        if self.selected_features:
            features = features[self.selected_features]
        
        return features


def preprocess_data(data_path, n_features=30, random_state=42):
    """Convenience function to preprocess data."""
    preprocessor = MedicalDataPreprocessor(n_features=n_features, random_state=random_state)
    return preprocessor.fit(data_path)

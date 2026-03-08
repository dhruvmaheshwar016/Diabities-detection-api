"""
Preprocessing utilities for disease prediction models.
Implements the same preprocessing pipeline used in model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DiabetesPreprocessor:
    """Preprocessing pipeline for diabetes prediction model."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False
        
    def fit(self, X_train_df):
        """
        Fit the preprocessor on training data.
        
        Args:
            X_train_df: Training DataFrame with all features
            
        Returns:
            self
        """
        # Store feature column names
        self.feature_columns = X_train_df.columns.tolist()
        
        # Fit imputer
        self.imputer.fit(X_train_df[[col for col in X_train_df.columns 
                                     if X_train_df[col].dtype in [np.number]]])
        
        # Fit scaler
        X_imputed = self.imputer.transform(X_train_df[[col for col in X_train_df.columns 
                                                       if X_train_df[col].dtype in [np.number]]])
        self.scaler.fit(X_imputed)
        
        self.is_fitted = True
        return self
    
    def transform(self, X_df):
        """
        Transform input data using fitted preprocessor.
        
        Args:
            X_df: Input DataFrame
            
        Returns:
            Preprocessed NumPy array ready for model prediction
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first. Call fit() on training data.")
        
        X_copy = X_df.copy()
        
        # Select numeric columns
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        # Impute missing values
        X_copy[numeric_cols] = self.imputer.transform(X_copy[numeric_cols])
        
        # Scale features
        X_scaled = self.scaler.transform(X_copy[numeric_cols])
        
        return X_scaled


class FeatureEngineer:
    """Creates engineered features for diabetes prediction."""
    
    @staticmethod
    def create_diabetes_features(df):
        """
        Create advanced diabetes-specific engineered features.
        
        Args:
            df: Input DataFrame with raw clinical features
            
        Returns:
            DataFrame with engineered features added
        """
        X = df.copy()
        
        # Core engineered features
        X['glucose_risk'] = X['glucose'] / 125  # Fasting glucose reference (125 mg/dL)
        X['glucose_bmi'] = X['glucose'] * X['bmi'] / 100  # Glucose-BMI interaction
        X['glucose_insulin_risk'] = (X['glucose'] * X['insulin']) / 50  # Glucose-insulin interaction
        X['lipid_risk'] = (X['triglycerides'] + X['cholesterol']) / (X['hdl'] + 1)  # Lipid risk composite
        X['cholesterol_risk'] = X['cholesterol'] / (X['hdl'] + 1)  # Cholesterol ratio
        X['ldl_risk'] = X['ldl'] / (X['hdl'] + 1)  # LDL/HDL ratio
        X['trigly_hdl'] = X['triglycerides'] / (X['hdl'] + 1)  # Triglyceride/HDL ratio
        X['bmi_category'] = pd.cut(X['bmi'], bins=[0, 25, 30, 100], labels=[0, 1, 2]).astype(int)  # BMI categories
        X['age_risk'] = X['age'] / 60  # Age normalization
        X['bp_risk'] = X['blood_pressure'] / 140  # Blood pressure normalization
        X['insulin_risk'] = np.log1p(X['insulin'] / 5)  # Log-transformed insulin risk
        X['metabolic_score'] = (X['glucose_risk'] + X['lipid_risk'] + X['bmi'] / 30 + X['insulin_risk']) / 4
        
        # Categorical encoding
        categorical_cols = ['gender', 'smoking', 'family_history']
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col + '_enc'] = le.fit_transform(X[col].astype(str))
        
        # Drop original categorical columns
        X = X.drop(columns=[col for col in categorical_cols if col in X.columns], errors='ignore')
        
        # Fill any remaining NaN values with median
        X = X.fillna(X.median())
        
        return X


def preprocess_prediction_input(data_dict, engineer_features=True):
    """
    Preprocess input data for prediction.
    
    Args:
        data_dict: Dictionary with input features
        engineer_features: Whether to apply feature engineering
        
    Returns:
        DataFrame ready for prediction
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([data_dict])
    
    if engineer_features:
        # Apply feature engineering
        df = FeatureEngineer.create_diabetes_features(df)
    
    # Handle missing values with median imputation (for individual samples)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

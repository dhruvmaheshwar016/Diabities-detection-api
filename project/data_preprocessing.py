import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def preprocess_data(data_path, n_features=50):
    """
    Preprocess the clinical data.

    Args:
        data_path (str): Path to the CSV file containing the data.
        n_features (int): Number of features to select using RFE.

    Returns:
        features (pd.DataFrame): Preprocessed features.
        targets (pd.DataFrame): Target variables.
        scaler (StandardScaler): Fitted scaler for numerical features.
        encoder (OneHotEncoder): Fitted encoder for categorical features.
        selected_features (list): List of selected feature names.
    """
    df = pd.read_csv(data_path)
    
    # Assume target columns are 'diabetes' and 'cad'
    targets = df[['diabetes', 'cad']]
    features = df.drop(['diabetes', 'cad'], axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols = features.select_dtypes(include=[np.number]).columns
    categorical_cols = features.select_dtypes(include=['object']).columns
    
    # KNN imputation for numerical features
    imputer = KNNImputer(n_neighbors=5)
    features[numerical_cols] = imputer.fit_transform(features[numerical_cols])
    
    # Standard scaling for numerical features
    scaler = StandardScaler()
    features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
    
    # One-hot encoding for categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(features[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Combine numerical and encoded categorical features
    features = pd.concat([features[numerical_cols], encoded_df], axis=1)
    
    # Feature selection using RFE with LogisticRegression
    estimator = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator, n_features_to_select=n_features)
    features_selected = rfe.fit_transform(features, targets['diabetes'])  # Using diabetes as the target for RFE
    selected_features = features.columns[rfe.support_]
    features = pd.DataFrame(features_selected, columns=selected_features)
    
    return features, targets, scaler, encoder, selected_features
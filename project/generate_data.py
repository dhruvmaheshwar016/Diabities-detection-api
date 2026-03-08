import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
data = {
    'age': np.random.randint(20, 80, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'bmi': np.random.normal(25, 5, n_samples),
    'glucose': np.random.normal(100, 20, n_samples),
    'blood_pressure': np.random.normal(120, 15, n_samples),
    'cholesterol': np.random.normal(200, 40, n_samples),
    'hdl': np.random.normal(50, 10, n_samples),
    'ldl': np.random.normal(130, 30, n_samples),
    'triglycerides': np.random.normal(150, 50, n_samples),
    'insulin': np.random.normal(10, 5, n_samples),
    'smoking': np.random.choice(['Yes', 'No'], n_samples),
    'family_history': np.random.choice(['Yes', 'No'], n_samples),
    'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'cad': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
}

# Introduce some missing values
for col in ['bmi', 'glucose', 'blood_pressure', 'cholesterol']:
    mask = np.random.choice([True, False], n_samples, p=[0.1, 0.9])
    data[col][mask] = np.nan

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/clinical_data.csv', index=False)

print("Sample dataset generated and saved to data/clinical_data.csv")
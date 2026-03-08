"""
Synthetic Clinical Dataset Generator for Multi-Disease Prediction

Generates a realistic 100,000 patient dataset with:
- 17 clinical and demographic features
- 2 disease targets (diabetes and CAD)
- Realistic feature distributions
- Proper medical correlations
- Balanced class distributions
"""

import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm, expon, bernoulli
import os

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("SYNTHETIC CLINICAL DATASET GENERATOR")
print("="*80)

# Set random seed for reproducibility
np.random.seed(42)

N_SAMPLES = 100000

print(f"\n[INFO] Generating {N_SAMPLES:,} patient records...")

# ============================================================================
# 1. DEMOGRAPHIC FEATURES
# ============================================================================
print("\n1. Generating demographic features...")

# Age: Right-skewed distribution (more older patients)
age = np.random.gamma(shape=9, scale=6.5, size=N_SAMPLES) + 20
age = np.clip(age, 18, 100).astype(int)

# Gender: Balanced
gender = np.random.choice(['Male', 'Female'], size=N_SAMPLES, p=[0.48, 0.52])

# Smoking status: Categorical with realistic prevalence
smoking = np.random.choice(['Never', 'Former', 'Current'], size=N_SAMPLES, 
                           p=[0.55, 0.30, 0.15])

# Physical activity: Categorical (hours per week)
physical_activity = np.random.choice([0, 1, 2, 3, 4, 5], size=N_SAMPLES,
                                    p=[0.15, 0.20, 0.30, 0.20, 0.10, 0.05])

# Family history
family_history_diabetes = np.random.choice([0, 1], size=N_SAMPLES, p=[0.65, 0.35])
family_history_cad = np.random.choice([0, 1], size=N_SAMPLES, p=[0.70, 0.30])

print(f"  [OK] Age: {age.min()}-{age.max()} years")
print(f"  [OK] Gender: {np.sum(gender=='Male')} Male, {np.sum(gender=='Female')} Female")
print(f"  [OK] Smoking: Never={np.sum(smoking=='Never')}, Former={np.sum(smoking=='Former')}, Current={np.sum(smoking=='Current')}")

# ============================================================================
# 2. ANTHROPOMETRIC FEATURES
# ============================================================================
print("\n2. Generating anthropometric features...")

# BMI: Age-dependent (increases with age)
age_factor = (age - age.min()) / (age.max() - age.min())
bmi_mean = 24 + 8 * age_factor  # Increases from 24 to 32 with age
bmi = np.random.normal(bmi_mean, 3.5, size=N_SAMPLES)
bmi = np.clip(bmi, 15, 55)

# Waist circumference: Correlated with BMI and gender
gender_factor = (gender == 'Male').astype(float)  # Males have larger waist
waist_circumference = 70 + 2.5 * bmi + 5 * gender_factor + np.random.normal(0, 3, size=N_SAMPLES)
waist_circumference = np.clip(waist_circumference, 50, 150)

print(f"  [OK] BMI: {bmi.mean():.1f} ± {bmi.std():.1f} (range: {bmi.min():.1f}-{bmi.max():.1f})")
print(f"  [OK] Waist circumference: {waist_circumference.mean():.1f} ± {waist_circumference.std():.1f} cm")

# ============================================================================
# 3. BLOOD PRESSURE FEATURES
# ============================================================================
print("\n3. Generating blood pressure features...")

# Systolic BP: Age-dependent and BMI-dependent
sbp_base = 110 + 0.5 * age + 0.8 * (bmi - 25)
systolic_bp = sbp_base + np.random.normal(0, 8, size=N_SAMPLES)
systolic_bp = np.clip(systolic_bp, 85, 220)

# Diastolic BP: Correlated with systolic
diastolic_bp = 0.5 * systolic_bp + np.random.normal(20, 5, size=N_SAMPLES)
diastolic_bp = np.clip(diastolic_bp, 55, 140)

print(f"  [OK] Systolic BP: {systolic_bp.mean():.1f} ± {systolic_bp.std():.1f} mmHg")
print(f"  [OK] Diastolic BP: {diastolic_bp.mean():.1f} ± {diastolic_bp.std():.1f} mmHg")

# ============================================================================
# 4. GLUCOSE AND GLYCEMIC CONTROL FEATURES
# ============================================================================
print("\n4. Generating glucose and glycemic control features...")

# Fasting glucose: Age, BMI, and family history dependent
# Higher values indicate diabetes risk
fasting_glucose_base = 90 + 0.3 * age + 1.2 * (bmi - 25) + 15 * family_history_diabetes
fasting_glucose = fasting_glucose_base + np.random.normal(0, 8, size=N_SAMPLES)
fasting_glucose = np.clip(fasting_glucose, 50, 400)

# HbA1c: Highly correlated with fasting glucose (with some noise)
hba1c = 4.5 + 0.028 * fasting_glucose + np.random.normal(0, 0.3, size=N_SAMPLES)
hba1c = np.clip(hba1c, 3.5, 13)

# Insulin level: Depends on glucose and BMI (insulin resistance)
# Higher insulin with obesity and high glucose
insulin_base = 5 + 0.02 * fasting_glucose + 0.5 * (bmi - 25) + 2 * family_history_diabetes
insulin_level = insulin_base + np.random.exponential(2, size=N_SAMPLES)
insulin_level = np.clip(insulin_level, 0.5, 50)

print(f"  [OK] Fasting glucose: {fasting_glucose.mean():.1f} ± {fasting_glucose.std():.1f} mg/dL")
print(f"  [OK] HbA1c: {hba1c.mean():.2f} ± {hba1c.std():.2f} %")
print(f"  [OK] Insulin: {insulin_level.mean():.2f} ± {insulin_level.std():.2f} mIU/L")

# ============================================================================
# 5. LIPID PROFILE FEATURES
# ============================================================================
print("\n5. Generating lipid profile features...")

# Total cholesterol: Age and family history dependent
cholesterol_total = 150 + 0.4 * age + 0.7 * (bmi - 25) + 20 * family_history_cad
cholesterol_total = cholesterol_total + np.random.normal(0, 15, size=N_SAMPLES)
cholesterol_total = np.clip(cholesterol_total, 100, 400)

# HDL (protective factor): Inverse correlation with obesity, age, and smoking
hdl_base = 55 - 0.15 * age - 1.2 * (bmi - 25) - 8 * (smoking == 'Current').astype(float)
hdl = hdl_base + np.random.normal(0, 8, size=N_SAMPLES)
hdl = np.clip(hdl, 20, 100)

# LDL: Calculated from cholesterol, HDL, and triglycerides
# Will be calculated after triglycerides

# Triglycerides: Correlated with glucose, BMI, and age
tg_base = 100 + 0.5 * fasting_glucose + 2 * (bmi - 25) + 0.2 * age
triglycerides = tg_base + np.random.exponential(20, size=N_SAMPLES)
triglycerides = np.clip(triglycerides, 30, 600)

# LDL: Using Friedewald formula (approximate)
# LDL = Total Chol - HDL - (TG / 5)
# Add some noise around the calculation
ldl_calculated = cholesterol_total - hdl - (triglycerides / 5)
ldl = ldl_calculated + np.random.normal(0, 10, size=N_SAMPLES)
ldl = np.clip(ldl, 20, 300)

print(f"  [OK] Total Cholesterol: {cholesterol_total.mean():.1f} ± {cholesterol_total.std():.1f} mg/dL")
print(f"  [OK] HDL: {hdl.mean():.1f} ± {hdl.std():.1f} mg/dL")
print(f"  [OK] LDL: {ldl.mean():.1f} ± {ldl.std():.1f} mg/dL")
print(f"  [OK] Triglycerides: {triglycerides.mean():.1f} ± {triglycerides.std():.1f} mg/dL")

# ============================================================================
# 6. DIABETES TARGET: Logistic model
# ============================================================================
print("\n6. Generating diabetes target variable...")

# Diabetes probability based on multiple risk factors
# Risk factors: glucose, HbA1c, BMI, age, family history, insulin

# Normalize features for logistic model
glucose_norm = (fasting_glucose - fasting_glucose.min()) / (fasting_glucose.max() - fasting_glucose.min())
hba1c_norm = (hba1c - hba1c.min()) / (hba1c.max() - hba1c.min())
bmi_norm = (bmi - bmi.min()) / (bmi.max() - bmi.min())
age_norm = (age - age.min()) / (age.max() - age.min())
insulin_norm = (insulin_level - insulin_level.min()) / (insulin_level.max() - insulin_level.min())

# Linear combination of risk factors (reduced coefficients for realistic prevalence ~12%)
logit_diabetes = (
    1.8 * glucose_norm +      # Glucose is strong predictor
    1.5 * hba1c_norm +        # HbA1c is strong predictor
    0.8 * bmi_norm +          # BMI increases risk
    0.5 * age_norm +          # Age increases risk
    1.0 * insulin_norm +      # Insulin resistance
    0.5 * family_history_diabetes +  # Family history
    -0.3 * physical_activity / 5 -   # Physical activity protects
    3.5  # Intercept to shift baseline probability down
)

# Convert logit to probability using sigmoid
prob_diabetes = 1 / (1 + np.exp(-logit_diabetes))

# Generate binary outcome based on probability
diabetes = (np.random.random(N_SAMPLES) < prob_diabetes).astype(int)

print(f"  [OK] Diabetes prevalence: {diabetes.mean()*100:.1f}% ({diabetes.sum():,} cases)")
print(f"  [OK] Diabetes probability range: {prob_diabetes.min():.3f} - {prob_diabetes.max():.3f}")

# ============================================================================
# 7. CAD TARGET: Logistic model
# ============================================================================
print("\n7. Generating CAD target variable...")

# CAD probability based on different risk factors
# Risk factors: LDL, HDL, blood pressure, age, family history, smoking

# Normalize lipids and BP
ldl_norm = (ldl - ldl.min()) / (ldl.max() - ldl.min())
hdl_norm = (hdl - hdl.min()) / (hdl.max() - hdl.min())
sbp_norm = (systolic_bp - systolic_bp.min()) / (systolic_bp.max() - systolic_bp.min())
smoking_current = (smoking == 'Current').astype(float)

# Linear combination of CAD risk factors (reduced for realistic prevalence ~7%)
logit_cad = (
    1.5 * ldl_norm +          # High LDL increases risk
    -1.2 * hdl_norm +         # Low HDL increases risk
    1.4 * sbp_norm +          # High blood pressure increases risk
    0.8 * age_norm +          # Age increases risk
    0.9 * family_history_cad + # Family history
    1.2 * smoking_current +    # Current smoking
    0.4 * diabetes +          # Diabetes increases CAD risk
    0.4 * bmi_norm -          # Obesity increases risk
    0.2 * physical_activity / 5 - # Physical activity protects
    3.8  # Intercept to shift baseline probability down
)

# Convert logit to probability
prob_cad = 1 / (1 + np.exp(-logit_cad))

# Generate binary outcome
cad = (np.random.random(N_SAMPLES) < prob_cad).astype(int)

print(f"  [OK] CAD prevalence: {cad.mean()*100:.1f}% ({cad.sum():,} cases)")
print(f"  [OK] CAD probability range: {prob_cad.min():.3f} - {prob_cad.max():.3f}")

# Comorbidity statistics
both_diseases = ((diabetes == 1) & (cad == 1)).sum()
print(f"  [OK] Both diseases: {both_diseases:,} ({both_diseases/N_SAMPLES*100:.1f}%)")

# ============================================================================
# 8. CREATE DATAFRAME AND SAVE
# ============================================================================
print("\n8. Creating and saving dataset...")

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'bmi': np.round(bmi, 2),
    'systolic_bp': np.round(systolic_bp, 1),
    'diastolic_bp': np.round(diastolic_bp, 1),
    'cholesterol_total': np.round(cholesterol_total, 1),
    'hdl': np.round(hdl, 1),
    'ldl': np.round(ldl, 1),
    'triglycerides': np.round(triglycerides, 1),
    'fasting_glucose': np.round(fasting_glucose, 1),
    'hba1c': np.round(hba1c, 2),
    'insulin_level': np.round(insulin_level, 2),
    'smoking_status': smoking,
    'physical_activity': physical_activity,
    'family_history_diabetes': family_history_diabetes,
    'family_history_cad': family_history_cad,
    'waist_circumference': np.round(waist_circumference, 1),
    'diabetes': diabetes,
    'cad': cad
})

print(f"\n[INFO] Dataset shape: {df.shape}")
print(f"\n[INFO] Dataset summary:")
print(df.describe())

# Create output directory
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = 'data/clinical_dataset_100k.csv'
df.to_csv(output_path, index=False)

print(f"\n[OK] Dataset saved to: {output_path}")
print(f"[OK] File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

# ============================================================================
# 9. DATA QUALITY CHECKS
# ============================================================================
print("\n9. Data quality validation...")

# Check for missing values
print(f"  [OK] Missing values: {df.isnull().sum().sum()}")

# Check for duplicates
print(f"  [OK] Duplicate rows: {df.duplicated().sum()}")

# Feature correlation analysis
print(f"\n[INFO] Feature-diabetes correlations:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in ['fasting_glucose', 'hba1c', 'bmi', 'insulin_level', 'cholesterol_total']:
    if col in df.columns:
        corr = df[col].corr(df['diabetes'])
        print(f"  - {col}: {corr:.4f}")

print(f"\n[INFO] Feature-CAD correlations:")
for col in ['ldl', 'hdl', 'systolic_bp', 'cholesterol_total', 'triglycerides']:
    if col in df.columns:
        corr = df[col].corr(df['cad'])
        print(f"  - {col}: {corr:.4f}")

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DATASET GENERATION COMPLETE")
print("="*80)

print(f"\nDataset Summary:")
print(f"  Total patients: {N_SAMPLES:,}")
print(f"  Total features: {len(df.columns) - 2} (+ 2 targets)")
print(f"\n  Diabetes cases: {diabetes.sum():,} ({diabetes.mean()*100:.1f}%)")
print(f"  CAD cases: {cad.sum():,} ({cad.mean()*100:.1f}%)")
print(f"  Both diseases: {both_diseases:,} ({both_diseases/N_SAMPLES*100:.1f}%)")
print(f"  Neither disease: {((diabetes==0) & (cad==0)).sum():,} ({((diabetes==0) & (cad==0)).mean()*100:.1f}%)")

print(f"\nFeature Ranges:")
for col in df.select_dtypes(include=[np.number]).columns:
    if col not in ['diabetes', 'cad']:
        print(f"  {col:25s}: {df[col].min():8.2f} - {df[col].max():8.2f}")

print("\n[OK] Ready for model training!")
print("="*80 + "\n")

# Multi-Disease Detection Pipeline

This project implements a hybrid CNN-Transformer deep learning model for early detection of Type 2 Diabetes Mellitus and Coronary Artery Disease using multimodal clinical data.

## Project Structure

```
project/
├── data/
│   └── clinical_data.csv  # Place your dataset here
├── data_preprocessing.py  # Data preprocessing pipeline
├── model.py               # PyTorch model implementation
├── train.py               # Training script
├── evaluate.py            # Evaluation and metrics computation
├── explainability.py      # SHAP explainability analysis
└── main.py                # Main script to run the pipeline
```

## Dataset Format

The dataset should be a CSV file named `clinical_data.csv` in the `data/` directory with the following structure:

- Columns: demographic attributes, laboratory results, clinical indicators, cardiovascular and metabolic risk factors
- Target columns: `diabetes` (0/1 for Type 2 Diabetes), `cad` (0/1 for Coronary Artery Disease)
- Mixed data types: numerical and categorical

## Installation

1. Ensure Python 3.7+ is installed.
2. Install dependencies:
   ```
   pip install torch scikit-learn pandas numpy matplotlib shap tqdm
   ```

## Usage

1. Place your dataset in `data/clinical_data.csv`.
2. Run the main script:
   ```
   python main.py
   ```

This will:
- Preprocess the data (imputation, scaling, encoding, feature selection)
- Train the hybrid CNN-Transformer model
- Evaluate performance with metrics and ROC curves
- Generate SHAP explanations for feature importance

## Model Architecture

- **CNN Component**: 1D convolutions for local feature extraction
- **Transformer Encoder**: Multi-head self-attention for contextual representations
- **Fusion**: Concatenation of CNN and Transformer features
- **Classification Head**: Dense layers with dropout for multi-label prediction

## Hyperparameters

- Batch size: 32
- Learning rate: 0.001
- Epochs: 50
- Features selected: 50 (configurable in `preprocess_data`)

## Output

- Training progress and validation loss
- Evaluation metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC) for each disease
- ROC curves visualization
- SHAP summary plots and feature importance bar charts

## Notes

- The model uses BCEWithLogitsLoss for multi-label classification
- Feature selection uses RFE with LogisticRegression
- SHAP explanations are computed on a subset of test data for efficiency
- Ensure your dataset has sufficient samples for training (recommend >1000 samples)
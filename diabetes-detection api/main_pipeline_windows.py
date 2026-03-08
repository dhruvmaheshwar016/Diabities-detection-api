"""
Main orchestration script for improved multi-disease prediction pipeline.
This is a Windows-compatible version with proper UTF-8 encoding and no Unicode characters in print statements.
"""

import os
import sys
import io

# Set UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent / 'training'))
sys.path.insert(0, str(Path(__file__).parent / 'evaluation'))
sys.path.insert(0, str(Path(__file__).parent / 'explainability'))

# Set matplotlib to use non-interactive backend before other imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_preprocessing_windows import MedicalDataPreprocessor
from training.train_models import MultiModelTrainer
from training.hyperparameter_tuning import HyperparameterOptimizer
from evaluation.evaluate_models import ModelEvaluator, PlotGenerator
from evaluation.plot_roc_curves import ROCCurveAnalyzer
from explainability.shap_analysis import CompoundExplainer

import warnings
warnings.filterwarnings('ignore')


class EnhancedPipelineOrchestrator:
    """Main orchestrator for the improved ML pipeline."""
    
    def __init__(self, data_path='data/clinical_data.csv', random_state=42, cv=5):
        """Initialize orchestrator."""
        self.data_path = data_path
        self.random_state = random_state
        self.cv = cv
        self.preprocessor = None
        self.trainer = None
        self.evaluator = None
        self.roc_analyzer = None
        
    def run_complete_pipeline(self):
        """Execute the complete improved pipeline."""
        
        print("\n" + "="*60)
        print("=  ENHANCED MULTI-DISEASE PREDICTION PIPELINE  =".center(60))
        print("="*60)
        
        # Step 1: Data Preprocessing
        self.step_preprocessing()
        
        # Step 2: Train Models with CV
        self.step_train_with_cv()
        
        # Step 3: Final Model Training
        self.step_final_training()
        
        # Step 4: Comprehensive Evaluation
        self.step_evaluation()
        
        # Step 5: Generate Plots
        self.step_generate_plots()
        
        # Step 6: SHAP Explainability
        self.step_explainability()
        
        print("\n" + "="*60)
        print("=  PIPELINE COMPLETED SUCCESSFULLY!  =".center(60))
        print("="*60 + "\n")
        
    def step_preprocessing(self):
        """Execute data preprocessing step."""
        print("\n" + "-"*60)
        print("STEP 1: DATA PREPROCESSING & FEATURE ENGINEERING")
        print("-"*60)
        
        # Load and preprocess
        self.preprocessor = MedicalDataPreprocessor(n_features=30, random_state=self.random_state)
        processed_data = self.preprocessor.fit(self.data_path)
        
        self.X_train = processed_data['X_train']
        self.y_diabetes = processed_data['y_diabetes_train']
        self.y_cad = processed_data['y_cad_train']
        self.feature_names = processed_data['feature_names']
        
        print(f"\n[OK] Preprocessing Complete:")
        print(f"     * Features: {self.X_train.shape[1]}")
        print(f"     * Samples: {self.X_train.shape[0]}")
        print(f"     * Data prepared and scaled")
        
        return processed_data
    
    def step_train_with_cv(self):
        """Execute training with cross-validation."""
        print("\n" + "-"*60)
        print("STEP 2: MODEL TRAINING WITH CROSS-VALIDATION")
        print("-"*60)
        
        self.trainer = MultiModelTrainer(random_state=self.random_state)
        self.trainer.create_models()
        
        # Train for Diabetes
        print("\n" + "="*60)
        print("DIABETES PREDICTION")
        print("="*60)
        self.trainer.train_with_cv(self.X_train, self.y_diabetes, cv=self.cv, 
                                  disease_name='Diabetes')
        
        # Train for CAD
        print("\n" + "="*60)
        print("CORONARY ARTERY DISEASE (CAD) PREDICTION")
        print("="*60)
        self.trainer.train_with_cv(self.X_train, self.y_cad, cv=self.cv,
                                  disease_name='CAD')
        
        return self.trainer
    
    def step_final_training(self):
        """Execute final model training on full dataset."""
        print("\n" + "-"*60)
        print("STEP 3: FINAL MODEL TRAINING")
        print("-"*60)
        
        # Train final models for both diseases
        self.trainer.train_final_models(self.X_train, self.y_diabetes, 
                                       disease_name='Diabetes')
        self.trainer.train_final_models(self.X_train, self.y_cad,
                                       disease_name='CAD')
        
        # Select best models
        self.trainer.select_best_model('Diabetes', metric='mean_auc')
        self.trainer.select_best_model('CAD', metric='mean_auc')
        
        return self.trainer
    
    def step_evaluation(self):
        """Execute comprehensive evaluation."""
        print("\n" + "-"*60)
        print("STEP 4: MODEL EVALUATION")
        print("-"*60)
        
        self.evaluator = ModelEvaluator()
        self.roc_analyzer = ROCCurveAnalyzer()
        
        # Get predictions from best models
        diabetes_model = self.trainer.trained_models['Diabetes'][self.trainer.best_model_name]
        cad_model = self.trainer.trained_models['CAD'][self.trainer.best_model_name]
        
        # Diabetes evaluation
        y_pred_diabetes = diabetes_model.predict(self.X_train)
        y_proba_diabetes = diabetes_model.predict_proba(self.X_train)
        
        self.evaluator.print_metrics(self.y_diabetes, y_pred_diabetes, 
                                    y_proba_diabetes, diabetes_model.__class__.__name__,
                                    'Diabetes')
        
        # CAD evaluation
        y_pred_cad = cad_model.predict(self.X_train)
        y_proba_cad = cad_model.predict_proba(self.X_train)
        
        self.evaluator.print_metrics(self.y_cad, y_pred_cad, y_proba_cad,
                                    cad_model.__class__.__name__, 'CAD')
        
        # Compute ROC curves
        self.roc_analyzer.compute_roc(self.y_diabetes, y_proba_diabetes,
                                     'Best Model', 'Diabetes')
        self.roc_analyzer.compute_roc(self.y_cad, y_proba_cad,
                                     'Best Model', 'CAD')
        
        self.roc_analyzer.print_auc_summary()
        
        return self.evaluator
    
    def step_generate_plots(self):
        """Generate evaluation plots."""
        print("\n" + "-"*60)
        print("STEP 5: GENERATING VISUALIZATION PLOTS")
        print("-"*60)
        
        os.makedirs('results', exist_ok=True)
        
        # ROC curves
        fig1 = self.roc_analyzer.plot_roc_by_disease('Diabetes')
        plt.savefig('results/roc_diabetes.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: results/roc_diabetes.png")
        plt.close(fig1)
        
        fig2 = self.roc_analyzer.plot_roc_by_disease('CAD')
        plt.savefig('results/roc_cad.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: results/roc_cad.png")
        plt.close(fig2)
        
        # AUC comparison
        fig3 = self.roc_analyzer.plot_auc_comparison()
        plt.savefig('results/auc_comparison.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: results/auc_comparison.png")
        plt.close(fig3)
        
        # CV results visualization
        cv_results = self.trainer.cv_results['Diabetes']
        fig4 = PlotGenerator.plot_metric_distributions(cv_results, 'Diabetes')
        plt.savefig('results/cv_analysis_diabetes.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: results/cv_analysis_diabetes.png")
        plt.close(fig4)
        
        print("\n[OK] All plots generated and saved in results/")
        
    def step_explainability(self):
        """Execute SHAP explainability analysis."""
        print("\n" + "-"*60)
        print("STEP 6: SHAP EXPLAINABILITY ANALYSIS")
        print("-"*60)
        
        # Get best models
        diabetes_model = self.trainer.trained_models['Diabetes'][self.trainer.best_model_name]
        cad_model = self.trainer.trained_models['CAD'][self.trainer.best_model_name]
        
        # Analyze Diabetes
        print("\n" + "="*60)
        print("DIABETES PREDICTION EXPLAINABILITY")
        print("="*60)
        
        explainer_diabetes = CompoundExplainer(diabetes_model.model, 
                                             self.X_train.values, 
                                             self.feature_names)
        explainer_diabetes.explain_all(self.X_train.values[:100], 'Diabetes')
        
        # Analyze CAD
        print("\n" + "="*60)
        print("CAD PREDICTION EXPLAINABILITY")
        print("="*60)
        
        explainer_cad = CompoundExplainer(cad_model.model,
                                         self.X_train.values,
                                         self.feature_names)
        explainer_cad.explain_all(self.X_train.values[:100], 'CAD')
        
        print("\n[OK] Explainability analysis complete")
        
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "="*60)
        print("FINAL SUMMARY REPORT")
        print("="*60)
        
        print("\n1. MODEL COMPARISON:")
        print("-"*60)
        results_dfs = self.trainer.get_results_dataframe()
        for disease, df in results_dfs.items():
            print(f"\n{disease}:")
            print(df.to_string())
        
        print("\n\n2. BEST MODELS SELECTED:")
        print("-"*60)
        print(f"Diabetes: {self.trainer.best_model_name}")
        print(f"CAD:      {self.trainer.best_model_name}")
        
        print("\n3. AUC TARGETS:")
        print("-"*60)
        auc_summary = self.roc_analyzer.get_auc_summary()
        print(auc_summary[['Disease', 'Model', 'AUC']].to_string())
        
        # Check if targets met
        for idx, row in auc_summary.iterrows():
            if row['AUC'] >= 0.80:
                status = "[TARGET MET]"
            elif row['AUC'] >= 0.75:
                status = "[CLOSE]"
            else:
                status = "[NEEDS IMPROVEMENT]"
            print(f"{row['Disease']:15s} / {row['Model']:12s}: {row['AUC']:.4f} {status}")


def main():
    """Main entry point."""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run pipeline
    orchestrator = EnhancedPipelineOrchestrator(
        data_path='data/clinical_data.csv',
        random_state=42,
        cv=5
    )
    
    # Execute complete pipeline
    orchestrator.run_complete_pipeline()
    
    # Generate summary
    orchestrator.generate_summary_report()


if __name__ == '__main__':
    main()

"""
Test client for Disease Prediction API
Demonstrates how to use the FastAPI endpoints
"""

import requests
import json
from typing import Dict, List

# Base URL for the API
BASE_URL = "http://localhost:8000"


class DiseasePredictionClient:
    """Client for interacting with the Disease Prediction API."""
    
    def __init__(self, base_url: str = BASE_URL):
        """Initialize the API client."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check if the API is healthy."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(self, patient_data: Dict) -> Dict:
        """
        Make a prediction for a single patient.
        
        Args:
            patient_data: Dictionary with patient clinical features
            
        Returns:
            Dictionary with predictions and risk classifications
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json=patient_data
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, patients: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple patients.
        
        Args:
            patients: List of dictionaries with patient clinical features
            
        Returns:
            List of dictionaries with predictions
        """
        response = self.session.post(
            f"{self.base_url}/predict-batch",
            json=patients
        )
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        response = self.session.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()
    
    def get_feature_info(self) -> Dict:
        """Get information about required features."""
        response = self.session.get(f"{self.base_url}/feature-info")
        response.raise_for_status()
        return response.json()


def test_single_prediction():
    """Test single patient prediction."""
    print("\n" + "="*60)
    print("TEST 1: Single Patient Prediction")
    print("="*60)
    
    client = DiseasePredictionClient()
    
    # Example patient data
    patient = {
        "age": 52,
        "gender": "M",
        "bmi": 29,
        "fasting_glucose": 130,
        "hba1c": 7.1,
        "cholesterol": 220,
        "ldl": 150,
        "hdl": 40,
        "triglycerides": 180,
        "blood_pressure": 140,
        "smoking": "yes",
        "family_history": "yes",
        "insulin": 12.5
    }
    
    try:
        prediction = client.predict(patient)
        print("\n✓ Prediction successful!")
        print(f"\nPatient Profile:")
        print(f"  Age: {patient['age']}, Gender: {patient['gender']}, BMI: {patient['bmi']}")
        print(f"  Glucose: {patient['fasting_glucose']} mg/dL, HbA1c: {patient['hba1c']}%")
        print(f"  HDL: {patient['hdl']}, LDL: {patient['ldl']}, Triglycerides: {patient['triglycerides']}")
        print(f"\nRisk Assessment:")
        print(f"  Diabetes Risk: {prediction['diabetes_risk_pct']:.1f}% - {prediction['diabetes_prediction'].upper()}")
        print(f"  CAD Risk: {prediction['cad_risk_pct']:.1f}% - {prediction['cad_prediction'].upper()}")
        print(f"  Model Confidence: {prediction['confidence'].upper()}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_batch_prediction():
    """Test batch patient predictions."""
    print("\n" + "="*60)
    print("TEST 2: Batch Predictions")
    print("="*60)
    
    client = DiseasePredictionClient()
    
    # Multiple patients
    patients = [
        {
            "age": 45,
            "gender": "F",
            "bmi": 24,
            "fasting_glucose": 100,
            "hba1c": 5.5,
            "cholesterol": 180,
            "ldl": 100,
            "hdl": 55,
            "triglycerides": 120,
            "blood_pressure": 120,
            "smoking": "no",
            "family_history": "no",
            "insulin": 5.0
        },
        {
            "age": 65,
            "gender": "M",
            "bmi": 32,
            "fasting_glucose": 160,
            "hba1c": 8.5,
            "cholesterol": 240,
            "ldl": 160,
            "hdl": 35,
            "triglycerides": 220,
            "blood_pressure": 160,
            "smoking": "yes",
            "family_history": "yes",
            "insulin": 18.0
        },
    ]
    
    try:
        predictions = client.predict_batch(patients)
        print(f"\n✓ Batch prediction successful for {len(predictions)} patients!")
        
        for i, pred in enumerate(predictions, 1):
            print(f"\nPatient {i}:")
            print(f"  Diabetes: {pred['diabetes_risk_pct']:.1f}% ({pred['diabetes_prediction']})")
            print(f"  CAD: {pred['cad_risk_pct']:.1f}% ({pred['cad_prediction']})")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_api_info():
    """Test API information endpoints."""
    print("\n" + "="*60)
    print("TEST 3: API Information")
    print("="*60)
    
    client = DiseasePredictionClient()
    
    try:
        # Health check
        health = client.health_check()
        print("\n✓ Health Check:")
        print(f"  Status: {health['status']}")
        print(f"  Diabetes Model: {health['models_loaded']['diabetes']}")
        print(f"  CAD Model: {health['models_loaded']['cad']}")
        
        # Model info
        model_info = client.get_model_info()
        print("\n✓ Model Information:")
        for disease, info in model_info.items():
            print(f"  {disease.upper()}: {info['type']} (Loaded: {info['loaded']})")
        
        # Feature info
        feature_info = client.get_feature_info()
        print("\n✓ Required Features:")
        for feature, details in list(feature_info['required_features'].items())[:5]:
            print(f"  - {feature}: {details['type']} ({details.get('unit', 'N/A')})")
        print(f"  ... and {len(feature_info['required_features']) - 5} more")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases & Error Handling")
    print("="*60)
    
    client = DiseasePredictionClient()
    
    # Test 1: Low-risk patient
    print("\nTest 4a: Low-risk patient")
    low_risk = {
        "age": 30,
        "gender": "F",
        "bmi": 21,
        "fasting_glucose": 90,
        "hba1c": 5.0,
        "cholesterol": 160,
        "ldl": 80,
        "hdl": 65,
        "triglycerides": 90,
        "blood_pressure": 110,
        "smoking": "no",
        "family_history": "no",
        "insulin": 4.0
    }
    
    try:
        pred = client.predict(low_risk)
        print(f"  ✓ Low-risk patient: Diabetes {pred['diabetes_risk_pct']:.1f}%, CAD {pred['cad_risk_pct']:.1f}%")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
    
    # Test 2: High-risk patient
    print("\nTest 4b: High-risk patient")
    high_risk = {
        "age": 75,
        "gender": "M",
        "bmi": 35,
        "fasting_glucose": 200,
        "hba1c": 9.5,
        "cholesterol": 280,
        "ldl": 190,
        "hdl": 30,
        "triglycerides": 300,
        "blood_pressure": 180,
        "smoking": "yes",
        "family_history": "yes",
        "insulin": 25.0
    }
    
    try:
        pred = client.predict(high_risk)
        print(f"  ✓ High-risk patient: Diabetes {pred['diabetes_risk_pct']:.1f}%, CAD {pred['cad_risk_pct']:.1f}%")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  Disease Prediction API - Test Client".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Run all tests
    test_single_prediction()
    test_batch_prediction()
    test_api_info()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")

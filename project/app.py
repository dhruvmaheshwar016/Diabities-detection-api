"""
FastAPI Backend for Disease Prediction Models
Provides REST API endpoints for CAD and Diabetes prediction
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import os
import numpy as np
import pandas as pd
import logging
from preprocessing import preprocess_prediction_input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App Setup
# ============================================================================
app = FastAPI(
    title="Disease Prediction API",
    description="API for CAD and Diabetes risk prediction using trained ML models",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Input/Output Models with Pydantic
# ============================================================================

class PatientData(BaseModel):
    """Input schema for patient data."""
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Gender: 'M' or 'F'")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    fasting_glucose: float = Field(..., ge=50, le=500, description="Fasting glucose in mg/dL")
    hba1c: float = Field(..., ge=3, le=15, description="HbA1c percentage")
    cholesterol: float = Field(..., ge=50, le=400, description="Total cholesterol in mg/dL")
    ldl: float = Field(..., ge=20, le=300, description="LDL cholesterol in mg/dL")
    hdl: float = Field(..., ge=10, le=200, description="HDL cholesterol in mg/dL")
    triglycerides: float = Field(..., ge=30, le=500, description="Triglycerides in mg/dL")
    blood_pressure: float = Field(..., ge=50, le=250, description="Systolic blood pressure")
    smoking: str = Field(default="no", description="Smoking status: 'yes' or 'no'")
    family_history: str = Field(default="no", description="Family history of disease: 'yes' or 'no'")
    insulin: float = Field(default=5.0, ge=0.1, le=500, description="Insulin level")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v.upper() not in ['M', 'F']:
            raise ValueError("Gender must be 'M' or 'F'")
        return v.upper()
    
    @validator('smoking', 'family_history')
    def validate_yes_no(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError("Value must be 'yes' or 'no'")
        return v.lower()


class PredictionResponse(BaseModel):
    """Output schema for predictions."""
    diabetes_risk: float = Field(..., ge=0, le=1, description="Diabetes risk probability (0-1)")
    diabetes_risk_pct: float = Field(..., description="Diabetes risk as percentage")
    diabetes_prediction: str = Field(..., description="Risk classification: 'low', 'moderate', 'high'")
    cad_risk: float = Field(default=None, description="CAD risk probability (0-1)")
    cad_risk_pct: float = Field(default=None, description="CAD risk as percentage")
    cad_prediction: str = Field(default=None, description="Risk classification: 'low', 'moderate', 'high'")
    confidence: str = Field(..., description="Model confidence level")
    
    class Config:
        schema_extra = {
            "example": {
                "diabetes_risk": 0.76,
                "diabetes_risk_pct": 76.0,
                "diabetes_prediction": "high",
                "cad_risk": 0.82,
                "cad_risk_pct": 82.0,
                "cad_prediction": "high",
                "confidence": "high"
            }
        }


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict
    version: str


# ============================================================================
# Model Loading
# ============================================================================

def load_models():
    """Load trained models from pickle files."""
    models = {}
    model_dir = 'models'
    
    # Try to load diabetes model
    diabetes_model_path = os.path.join(model_dir, 'diabetes_best_model_final.pkl')
    if os.path.exists(diabetes_model_path):
        try:
            with open(diabetes_model_path, 'rb') as f:
                models['diabetes'] = pickle.load(f)
            logger.info(f"✓ Loaded diabetes model from {diabetes_model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load diabetes model: {str(e)}")
    else:
        logger.warning(f"⚠ Diabetes model not found at {diabetes_model_path}")
    
    # Try to load CAD model (common alternative names)
    cad_paths = [
        os.path.join(model_dir, 'cad_best_model.pkl'),
        os.path.join(model_dir, 'cad_model.pkl'),
        os.path.join(model_dir, 'diabetes_best_model_final.pkl'),  # Using diabetes as fallback
    ]
    
    for cad_path in cad_paths:
        if os.path.exists(cad_path) and 'cad' not in models:
            try:
                with open(cad_path, 'rb') as f:
                    models['cad'] = pickle.load(f)
                logger.info(f"✓ Loaded CAD model from {cad_path}")
                break
            except Exception as e:
                logger.error(f"❌ Failed to load CAD model from {cad_path}: {str(e)}")
    
    if 'cad' not in models:
        logger.warning("⚠ CAD model not found. Using diabetes model for both predictions.")
        if 'diabetes' in models:
            models['cad'] = models['diabetes']
    
    return models


# Load models at startup
MODELS = load_models()

# ============================================================================
# Helper Functions
# ============================================================================

def classify_risk(probability, thresholds=None):
    """
    Classify risk level based on probability.
    
    Args:
        probability: Probability value (0-1)
        thresholds: Dict with 'low' and 'moderate' thresholds
        
    Returns:
        Risk classification string
    """
    if thresholds is None:
        thresholds = {'low': 0.33, 'moderate': 0.67}
    
    if probability <= thresholds['low']:
        return 'low'
    elif probability <= thresholds['moderate']:
        return 'moderate'
    else:
        return 'high'


def get_confidence_level(probability):
    """
    Determine model confidence based on probability distance from 0.5.
    
    Args:
        probability: Model probability output
        
    Returns:
        Confidence level string
    """
    distance = abs(probability - 0.5)
    if distance > 0.4:
        return 'high'
    elif distance > 0.25:
        return 'moderate'
    else:
        return 'low'


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Disease Prediction API",
        "version": "1.0.0",
        "description": "API for CAD and Diabetes risk prediction",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns status of API and loaded models.
    """
    return HealthCheck(
        status="healthy" if MODELS else "unhealthy",
        models_loaded={
            "diabetes": "loaded" if "diabetes" in MODELS else "not loaded",
            "cad": "loaded" if "cad" in MODELS else "not loaded"
        },
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_disease_risk(patient: PatientData):
    """
    Predict disease risk for a patient.
    
    Accepts patient clinical features and returns probability estimates
    for diabetes and CAD risk, along with risk classifications.
    
    Args:
        patient: PatientData model with clinical features
        
    Returns:
        PredictionResponse with risk probabilities and classifications
        
    Raises:
        HTTPException: If models are not loaded or prediction fails
    """
    
    if not MODELS:
        logger.error("No models loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded. Please check server logs."
        )
    
    try:
        # Convert pydantic model to dict
        patient_dict = patient.dict()
        
        logger.info(f"Processing prediction request for patient: age={patient.age}, bmi={patient.bmi}")
        
        # Preprocess input data (applies feature engineering)
        X_processed = preprocess_prediction_input(patient_dict, engineer_features=True)
        
        # Get feature columns that match model training
        feature_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        X_array = X_processed[feature_cols].values
        
        # Initialize prediction values
        diabetes_prob = 0.0
        cad_prob = 0.0
        
        # Get diabetes prediction
        if "diabetes" in MODELS:
            try:
                diabetes_prob = MODELS["diabetes"].predict_proba(X_array)[0, 1]
                logger.info(f"Diabetes prediction: {diabetes_prob:.4f}")
            except Exception as e:
                logger.error(f"Error in diabetes prediction: {str(e)}")
                diabetes_prob = 0.5  # Default to neutral probability on error
        
        # Get CAD prediction
        if "cad" in MODELS:
            try:
                cad_prob = MODELS["cad"].predict_proba(X_array)[0, 1]
                logger.info(f"CAD prediction: {cad_prob:.4f}")
            except Exception as e:
                logger.error(f"Error in CAD prediction: {str(e)}")
                cad_prob = 0.5  # Default to neutral probability on error
        
        # Classify risks
        diabetes_classification = classify_risk(diabetes_prob)
        cad_classification = classify_risk(cad_prob)
        
        # Determine confidence
        confidence = get_confidence_level(diabetes_prob)
        
        # Create response
        response = PredictionResponse(
            diabetes_risk=round(diabetes_prob, 4),
            diabetes_risk_pct=round(diabetes_prob * 100, 2),
            diabetes_prediction=diabetes_classification,
            cad_risk=round(cad_prob, 4) if cad_prob else None,
            cad_risk_pct=round(cad_prob * 100, 2) if cad_prob else None,
            cad_prediction=cad_classification if cad_prob else None,
            confidence=confidence
        )
        
        logger.info(f"✓ Prediction successful: diabetes={diabetes_classification} ({diabetes_prob:.2%}), "
                   f"cad={cad_classification} ({cad_prob:.2%})")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Data validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch", tags=["Predictions"])
async def predict_batch(patients: list[PatientData]):
    """
    Predict disease risk for multiple patients (batch processing).
    
    Args:
        patients: List of PatientData models
        
    Returns:
        List of PredictionResponse objects
    """
    predictions = []
    for patient in patients:
        prediction = await predict_disease_risk(patient)
        predictions.append(prediction)
    
    logger.info(f"✓ Batch prediction completed for {len(predictions)} patients")
    return predictions


@app.get("/model-info", tags=["Information"])
async def get_model_info():
    """
    Get information about loaded models.
    
    Returns model types and performance details if available.
    """
    info = {
        "diabetes_model": {
            "loaded": "diabetes" in MODELS,
            "type": type(MODELS.get("diabetes")).__name__ if "diabetes" in MODELS else None,
            "description": "Random Forest classifier for diabetes prediction (ROC-AUC: 0.8302)"
        },
        "cad_model": {
            "loaded": "cad" in MODELS,
            "type": type(MODELS.get("cad")).__name__ if "cad" in MODELS else None,
            "description": "Ensemble model for CAD prediction"
        }
    }
    return info


@app.get("/feature-info", tags=["Information"])
async def get_feature_info():
    """
    Get information about required features and their constraints.
    
    Returns schema of expected input features.
    """
    return {
        "required_features": {
            "age": {"type": "float", "min": 0, "max": 120, "unit": "years"},
            "gender": {"type": "str", "values": ["M", "F"]},
            "bmi": {"type": "float", "min": 10, "max": 60, "unit": "kg/m²"},
            "fasting_glucose": {"type": "float", "min": 50, "max": 500, "unit": "mg/dL"},
            "hba1c": {"type": "float", "min": 3, "max": 15, "unit": "%"},
            "cholesterol": {"type": "float", "min": 50, "max": 400, "unit": "mg/dL"},
            "ldl": {"type": "float", "min": 20, "max": 300, "unit": "mg/dL"},
            "hdl": {"type": "float", "min": 10, "max": 200, "unit": "mg/dL"},
            "triglycerides": {"type": "float", "min": 30, "max": 500, "unit": "mg/dL"},
            "blood_pressure": {"type": "float", "min": 50, "max": 250, "unit": "mmHg"},
        },
        "optional_features": {
            "smoking": {"type": "str", "values": ["yes", "no"], "default": "no"},
            "family_history": {"type": "str", "values": ["yes", "no"], "default": "no"},
            "insulin": {"type": "float", "min": 0.1, "max": 500, "default": 5.0, "unit": "mU/L"}
        },
        "engineered_features": [
            "glucose_risk",
            "glucose_bmi",
            "glucose_insulin_risk",
            "lipid_risk",
            "cholesterol_risk",
            "ldl_risk",
            "trigly_hdl",
            "bmi_category",
            "age_risk",
            "bp_risk",
            "insulin_risk",
            "metabolic_score"
        ]
    }


if __name__ == "__main__":
    # This won't run with 'uvicorn app:app', but useful for testing
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

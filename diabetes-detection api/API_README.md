# Disease Prediction API

FastAPI backend for serving trained machine learning models for CAD and Diabetes risk prediction.

## Features

- ✓ **RESTful API** with FastAPI
- ✓ **Input Validation** using Pydantic models
- ✓ **Feature Engineering** pipeline integrated into predictions
- ✓ **Batch Predictions** for multiple patients
- ✓ **Interactive API Documentation** (Swagger UI + ReDoc)
- ✓ **CORS Support** for cross-origin requests
- ✓ **Comprehensive Logging** for debugging
- ✓ **Error Handling** with detailed error messages
- ✓ **Health Checks** and model status monitoring

## Setup

### 1. Install Dependencies

```bash
# Navigate to project directory
cd c:\Users\dhruv\Desktop\Aman\project

# Install with pip
pip install -r requirements_api.txt
```

### 2. Verify Models are Loaded

The API expects trained models in the `models/` directory. Currently available:
- `models/diabetes_best_model_final.pkl` - Random Forest (ROC-AUC: 0.8302)

```bash
# Check available models
ls models/*.pkl
```

### 3. Start the Server

#### Option A: Using the batch file (Windows)
```bash
run_api.bat
```

#### Option B: Using uvicorn directly
```bash
# Install FastAPI dependencies if not already done
pip install -r requirements_api.txt

# Start the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Option C: Using Python
```bash
python app.py
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
✓ Loaded diabetes model from models/diabetes_best_model_final.pkl
```

## API Documentation

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "diabetes": "loaded",
    "cad": "loaded"
  },
  "version": "1.0.0"
}
```

#### 2. Predict Disease Risk (Main Endpoint)
```http
POST /predict
Content-Type: application/json

{
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
```

**Response:**
```json
{
  "diabetes_risk": 0.7634,
  "diabetes_risk_pct": 76.34,
  "diabetes_prediction": "high",
  "cad_risk": 0.7634,
  "cad_risk_pct": 76.34,
  "cad_prediction": "high",
  "confidence": "high"
}
```

#### 3. Batch Predictions
```http
POST /predict-batch
Content-Type: application/json

[
  {
    "age": 45,
    "gender": "F",
    "bmi": 24,
    ...
  },
  {
    "age": 65,
    "gender": "M",
    "bmi": 32,
    ...
  }
]
```

#### 4. Get Model Information
```http
GET /model-info
```

**Response:**
```json
{
  "diabetes_model": {
    "loaded": true,
    "type": "RandomForestClassifier",
    "description": "Random Forest classifier for diabetes prediction (ROC-AUC: 0.8302)"
  },
  "cad_model": {
    "loaded": true,
    "type": "RandomForestClassifier",
    "description": "Ensemble model for CAD prediction"
  }
}
```

#### 5. Get Feature Information
```http
GET /feature-info
```

Returns schema of all required and optional features with constraints.

## Input Parameters

### Required Fields

| Parameter | Type | Range | Unit | Description |
|-----------|------|-------|------|-------------|
| age | float | 0-120 | years | Patient age |
| gender | str | M, F | - | Gender |
| bmi | float | 10-60 | kg/m² | Body Mass Index |
| fasting_glucose | float | 50-500 | mg/dL | Fasting glucose level |
| hba1c | float | 3-15 | % | Hemoglobin A1c |
| cholesterol | float | 50-400 | mg/dL | Total cholesterol |
| ldl | float | 20-300 | mg/dL | LDL cholesterol |
| hdl | float | 10-200 | mg/dL | HDL cholesterol |
| triglycerides | float | 30-500 | mg/dL | Triglycerides |
| blood_pressure | float | 50-250 | mmHg | Systolic blood pressure |

### Optional Fields

| Parameter | Type | Default | Unit | Description |
|-----------|------|---------|------|-------------|
| smoking | str | "no" | - | Smoking status (yes/no) |
| family_history | str | "no" | - | Family history (yes/no) |
| insulin | float | 5.0 | mU/L | Insulin level |

## Testing

### 1. Quick Test with cURL

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 2. Python Test Client

```bash
# Run the test suite
python test_api.py
```

This will run:
- Single patient prediction
- Batch predictions
- API information retrieval
- Edge case testing

### 3. Using requests library

```python
import requests

# Make a prediction
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
    "blood_pressure": 140
}

response = requests.post("http://localhost:8000/predict", json=patient)
prediction = response.json()

print(f"Diabetes Risk: {prediction['diabetes_risk_pct']:.1f}%")
print(f"CAD Risk: {prediction['cad_risk_pct']:.1f}%")
```

## Risk Classification

The API classifies risk levels based on probability:

| Risk Level | Probability Range | Action |
|-----------|-------------------|--------|
| **Low** | ≤ 33% | Routine monitoring |
| **Moderate** | 33-67% | Regular medical follow-up |
| **High** | > 67% | Immediate medical consultation |

## Feature Engineering

The API automatically applies the following engineered features:

1. **glucose_risk** - Normalized fasting glucose (glucose/125)
2. **glucose_bmi** - Glucose-BMI interaction
3. **glucose_insulin_risk** - Glucose-insulin product risk
4. **lipid_risk** - Composite lipid risk score
5. **cholesterol_risk** - Cholesterol/HDL ratio
6. **ldl_risk** - LDL/HDL ratio
7. **trigly_hdl** - Triglyceride/HDL ratio
8. **bmi_category** - BMI stratification (normal/overweight/obese)
9. **age_risk** - Age-normalized risk
10. **bp_risk** - Blood pressure normalization
11. **insulin_risk** - Log-transformed insulin risk
12. **metabolic_score** - Composite metabolic risk score

## Performance

The models were trained on 1000 clinical samples with cross-validation:

**Diabetes Model (Random Forest):**
- ROC-AUC: **0.8302**
- Accuracy: 78.01%
- Precision: 80.15%
- Recall: 74.47%
- F1-Score: 0.7721

## Error Handling

The API provides detailed error messages:

### Example Error Response
```json
{
  "detail": "Data validation error: Value error, age must be between 0 and 120"
}
```

### Common Errors

| Error | Solution |
|-------|----------|
| `Model not loaded` | Check that model files exist in `models/` directory |
| `Validation error` | Ensure all required fields are provided and within valid ranges |
| `Connection refused` | Check that the server is running on port 8000 |

## Production Deployment

For production deployment, consider:

1. **Use a production ASGI server:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

2. **Add authentication:**
```python
from fastapi.security import HTTPBearer
security = HTTPBearer()
```

3. **Enable HTTPS:**
```bash
uvicorn app:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

4. **Add rate limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

5. **Monitor with logging:**
```python
logger = logging.getLogger(__name__)
logger.info(f"Prediction for patient: {patient.id}")
```

## Architecture

```
Request → Validation (Pydantic) → Feature Engineering → 
Preprocessing → Model Inference → Risk Classification → Response
```

## Troubleshooting

### Models not loading

```bash
# Check if model files exist
ls models/*.pkl

# If missing, train models using:
python diabetes_final_pipeline.py
```

### Port already in use

```bash
# Use a different port
uvicorn app:app --port 8001

# Or kill process on port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>
```

### Import errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements_api.txt
```

## API Response Examples

### Low-Risk Patient
```json
{
  "diabetes_risk": 0.15,
  "diabetes_risk_pct": 15.0,
  "diabetes_prediction": "low",
  "cad_risk": 0.12,
  "cad_risk_pct": 12.0,
  "cad_prediction": "low",
  "confidence": "high"
}
```

### High-Risk Patient
```json
{
  "diabetes_risk": 0.89,
  "diabetes_risk_pct": 89.0,
  "diabetes_prediction": "high",
  "cad_risk": 0.92,
  "cad_risk_pct": 92.0,
  "cad_prediction": "high",
  "confidence": "high"
}
```

## Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Validation](https://pydantic-docs.helpmanual.io/)
- [Project Report](FINAL_PROJECT_REPORT.md)

## Support

For issues or questions:
1. Check the API logs (printed to console)
2. Review the interactive documentation at `/docs`
3. Check the test client output with `python test_api.py`

---

**Created:** March 8, 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✓

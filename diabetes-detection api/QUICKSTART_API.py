"""
Quick Start Guide for FastAPI Disease Prediction Backend
"""

# 1. INSTALLATION
# ===============

## In PowerShell or Command Prompt:

```powershell
cd c:\Users\dhruv\Desktop\Aman\project

# Install dependencies
pip install -r requirements_api.txt
# or
pip install fastapi uvicorn pydantic requests

```

# 2. START THE SERVER
# ====================

## Option A: Simplest method
```powershell
python app.py
```

## Option B: Using uvicorn directly
```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Option C: Using batch file
```powershell
.\run_api.bat
```

## Expected Output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
✓ Loaded diabetes model from models/diabetes_best_model_final.pkl
```

# 3. ACCESS THE API
# ==================

Once running, open in browser:
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

# 4. MAKE PREDICTIONS
# ====================

## Using Python:
```python
import requests

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

response = requests.post("http://localhost:8000/predict", json=patient)
print(response.json())
```

## Using cURL:
```bash
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
    "blood_pressure": 140
  }'
```

# 5. RUN TESTS
# =============

```powershell
python test_api.py
```

This will test:
- Single predictions
- Batch predictions
- API endpoints
- Error handling

# 6. EXPECTED RESPONSE
# =====================

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

# 7. TROUBLESHOOTING
# ===================

## Error: "No module named uvicorn"
```powershell
pip install uvicorn
```

## Error: "Connection refused"
Make sure the server is running on port 8000

## Error: "Model not loaded"
Check that models/diabetes_best_model_final.pkl exists

## Port 8000 already in use
```powershell
# Use a different port
uvicorn app:app --port 8001
```

# 8. API ENDPOINTS
# =================

GET /
  └─ API information

GET /health
  └─ Health check

POST /predict
  └─ Single patient prediction

POST /predict-batch
  └─ Batch predictions

GET /model-info
  └─ Model information

GET /feature-info
  └─ Feature requirements

# 9. DEPLOYMENT
# ==============

For production use Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

With SSL/HTTPS:
```bash
uvicorn app:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```
"""

if __name__ == "__main__":
    print(__doc__)

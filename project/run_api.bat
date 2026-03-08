@echo off
REM FastAPI Server Launcher for Disease Prediction
REM Usage: run_api.bat

echo.
echo ========================================
echo  Disease Prediction API Server
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies if needed
echo Installing dependencies...
pip install -q -r requirements_api.txt

REM Start the API server
echo.
echo Starting FastAPI server on http://localhost:8000
echo.
echo Available endpoints:
echo   - API Docs:  http://localhost:8000/docs
echo   - ReDoc:     http://localhost:8000/redoc
echo   - Predict:   POST http://localhost:8000/predict
echo   - Health:    GET http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

uvicorn app:app --reload --host 0.0.0.0 --port 8000

pause

@echo off
echo Starting FastAPI Server and Streamlit Dashboard...
echo.

echo Starting FastAPI Server on http://localhost:8000
start "FastAPI Server" python fastapi_backend.py

timeout /t 3 /nobreak >nul

echo Starting Streamlit Dashboard on http://localhost:8501
start "Streamlit Dashboard" streamlit run streamlit_dashboard.py

echo.
echo Both services are starting...
echo - FastAPI Server: http://localhost:8000
echo - Streamlit Dashboard: http://localhost:8501
echo.
pause
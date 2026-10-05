@echo off
title AeroScan MRO Inspection HUD
echo ========================================================
echo   AEROSCAN: Multimodal Edge AI Defect Detection HUD
echo   Team AeroNauts - Tata Technologies InnoVent 2026-27
echo ========================================================
echo.
echo Installing/Verifying dependencies...
pip install -r requirements.txt
echo.
echo Launching Streamlit MRO HUD on localhost:8501...
python -m streamlit run dashboard.py
pause

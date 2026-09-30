@echo off
echo ==========================================
echo NADI-VALUE - Snapdragon Windows Build
echo ==========================================
python src\qnn_provider_check.py
echo.
python src\app_onnx.py
pause

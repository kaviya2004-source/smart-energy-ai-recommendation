@echo off
cd /d "%~dp0"
echo Creating person-detection prediction images from the saved YOLOv8 model...
.\.venv\Scripts\python.exe src\predict_yolo.py
pause

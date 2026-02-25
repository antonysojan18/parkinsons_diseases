@echo off
echo Starting Parkinson's Disease Detection App...
echo.

REM Go to project directory
cd /d "D:\parkinsons diseases"

REM Activate virtual environment (ONLY if you have one)
REM call venv\Scripts\activate

REM Run visualizer (optional – comment out if not needed every time)
python visualizer.py

REM Run Flask app
python app.py

pause

@echo off
REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the Flask app
echo.
echo Starting Cardiovascular Risk Classifier Web App...
echo.
echo The app will be available at: http://localhost:5000
echo.
echo Press CTRL+C to stop the server.
echo.

python app.py
pause

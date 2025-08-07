@echo off
echo Starting Twitter Sentiment Analysis Dashboard with DARK THEME...
echo.

REM Set environment variables for better performance and dark theme
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
set STREAMLIT_SERVER_MAX_MESSAGE_SIZE=1000
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_THEME_BASE=dark

REM Navigate to project directory
cd /d "c:\Users\ommou\OneDrive\Desktop\Twitter sentiment analysis\RealTimeSentimentApp"

REM Check if GPU is available
echo Checking GPU availability...
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo.

REM Kill any existing Streamlit processes on port 8502
echo Stopping any existing Streamlit processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8502') do taskkill /f /pid %%a >nul 2>&1
timeout /t 2 /nobreak >nul

REM Start Streamlit with dark theme and optimized settings
echo Starting DARK THEME Streamlit dashboard on http://localhost:8502
echo.
echo IMPORTANT NOTES:
echo - Dashboard now uses DARK THEME for better visual experience
echo - For large datasets (10K+ tweets), processing may take several minutes
echo - GPU acceleration is enabled for faster processing
echo - The browser may show "Please wait..." during long operations - this is normal
echo - DO NOT close this window while processing large datasets
echo.

streamlit run app/streamlit_app.py --server.port 8502 --server.headless false --server.fileWatcherType none --server.runOnSave false

pause

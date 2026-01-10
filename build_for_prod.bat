@echo off
echo ===========================================
echo      PetGuard Pro - Production Builder
echo ===========================================

echo [1/4] Installing Backend Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/4] Building Frontend (React)...
cd frontend
call npm install
call npm run build
if %errorlevel% neq 0 exit /b %errorlevel%
cd ..

echo [3/4] Moving Build to Backend...
if exist "backend\static" rmdir /s /q "backend\static"
mkdir "backend\static"
xcopy /e /i /y "frontend\dist\*" "backend\static"

echo [4/4] Starting Unified Server...
echo.
echo ----------------------------------------------------
echo    App is running at: http://localhost:8000
echo    (Press Ctrl+C to stop)
echo ----------------------------------------------------
echo.

set GEMINI_API_KEY=AIzaSyDJCJhY_ltyqQA7UuLiDcMJa8y29v9AO4g
python backend/production.py

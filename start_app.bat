@echo off
echo Starting Pet Detector...
set GEMINI_API_KEY=AIzaSyDJCJhY_ltyqQA7UuLiDcMJa8y29v9AO4g


start "Backend Server" cmd /k "cd backend && uvicorn main:app --reload --port 8000"
echo Backend started on port 8000.

start "Frontend Server" cmd /k "cd frontend && npm run dev"
echo Frontend started.

echo All systems go!

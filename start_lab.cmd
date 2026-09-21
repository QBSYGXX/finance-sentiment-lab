@echo off
cd /d "%~dp0"
set "LAB_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "LAB_PYTHON=.venv\Scripts\python.exe"
echo Local URL: http://127.0.0.1:8787
"%LAB_PYTHON%" -u -X utf8 run.py serve --port 8787
set "LAB_EXIT=%ERRORLEVEL%"
pause
exit /b %LAB_EXIT%

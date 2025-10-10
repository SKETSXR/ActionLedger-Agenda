@echo off
REM ===== config =====
set "CONDA_BAT=C:\Users\akshivk\AppData\Local\anaconda3\condabin\conda.bat"
set "ENV_NAME=best_all_env"
set "PROJECT_DIR=C:\Users\akshivk\Desktop\Action Ledger - Agenda"

REM ===== activate env =====
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo [ERROR] Could not activate "%ENV_NAME%".
  echo Check the name with:  conda env list
  pause
  exit /b 1
)

REM ===== go to project and run =====
cd /d "%PROJECT_DIR%"
python app.py

pause

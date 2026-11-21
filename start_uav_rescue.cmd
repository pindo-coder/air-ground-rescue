@echo off
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" tools\run_uav_rescue.py --profile full --detector yolo --planner igha_vctp --visible --realtime --keep-open %*

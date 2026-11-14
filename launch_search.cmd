@echo off
setlocal
if exist "%~dp0search.local.cmd" call "%~dp0search.local.cmd"
if not defined SEARCH_PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "SEARCH_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined SEARCH_PYTHON set "SEARCH_PYTHON=python"
"%SEARCH_PYTHON%" "%~dp0tools\run_search.py" --visible --realtime --keep-open %*
if errorlevel 1 pause

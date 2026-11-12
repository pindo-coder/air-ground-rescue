@echo off
setlocal
set "WORLD=B_washout.wbt"
if /I "%~1"=="A" set "WORLD=A_trail.wbt"
if /I "%~1"=="C" set "WORLD=C_generalization.wbt"
if exist "%~dp0webots.local.cmd" call "%~dp0webots.local.cmd"
if defined WEBOTS_EXE if not exist "%WEBOTS_EXE%" set "WEBOTS_EXE="
for %%E in (webotsw.exe webots.exe) do (
 if not defined WEBOTS_EXE if defined WEBOTS_HOME if exist "%WEBOTS_HOME%\msys64\mingw64\bin\%%E" set "WEBOTS_EXE=%WEBOTS_HOME%\msys64\mingw64\bin\%%E"
 if not defined WEBOTS_EXE if exist "%ProgramFiles%\Webots\msys64\mingw64\bin\%%E" set "WEBOTS_EXE=%ProgramFiles%\Webots\msys64\mingw64\bin\%%E"
)
for %%E in (webotsw.exe webots.exe) do if not defined WEBOTS_EXE if not "%%~$PATH:E"=="" set "WEBOTS_EXE=%%~$PATH:E"
if not defined WEBOTS_EXE (
 echo Webots R2025a was not found. Set WEBOTS_HOME or copy webots.local.cmd.example to webots.local.cmd.
 echo You can also open a worlds/*.wbt file manually in Webots.
 pause
 exit /b 1
)
start "" "%WEBOTS_EXE%" --batch --mode=realtime "%~dp0worlds\%WORLD%"

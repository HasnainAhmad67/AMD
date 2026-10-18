@echo off
rem Telltale helper for Windows.
rem   Double-click this file  -> menu
rem   .\telltale start        -> start the app (works in PowerShell and cmd)
rem   .\telltale help         -> all commands
rem The real logic lives in scripts\telltale.ps1.
setlocal
if not "%~1"=="" goto run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\telltale.ps1" menu
echo.
pause
exit /b 0
:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\telltale.ps1" %*
exit /b %ERRORLEVEL%

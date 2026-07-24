@echo off
rem Launch separately so this batch file can be replaced while the update runs.
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0bin\update.ps1" -WaitForInput
exit /b

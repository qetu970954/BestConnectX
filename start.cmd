@echo off
cd /d "%~dp0"
uv run python dashboard.py --open-browser %*
if errorlevel 1 pause

@echo off
cd /d "C:\Users\Personal\Desktop\Portfolio_ML"
title Start Portfolio
echo Portfolio is starting.
echo Close this window to stop the dashboard and market-data updates.
echo.
".venv\Scripts\python.exe" -m app.launcher
echo.
echo Portfolio stopped.
pause

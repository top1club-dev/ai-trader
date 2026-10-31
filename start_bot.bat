@echo off
cd /d "%~dp0"
echo Bot shuru ho raha hai...
:loop
python bot.py
echo.
echo Bot band hua. 30 second mein wapas chalu hoga... (rokne ke liye Ctrl+C dabao, phir Y)
timeout /t 30 /nobreak
goto loop

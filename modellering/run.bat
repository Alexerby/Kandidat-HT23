@echo off

set python_fil=.\main.py

if exist "%python_fil%" (
    rem Kör Python-filen
    python "%python_fil%"
) else (
    echo Fel: Python-filen hittades inte.
)

pause.


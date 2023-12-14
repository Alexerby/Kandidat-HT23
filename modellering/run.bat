@echo off

:: Prompta användaren för att bekräfta aktivering av den virtuella miljön
echo -----------------------------------------------
echo Vill du aktivera den virtuella miljön för att köra skriptet?
set /p activate_choice=JA/NEJ: 

if /i "%activate_choice%"=="JA" (
    echo -----------------------------------------------
    echo Aktiverar den virtuella miljön...
    call .\venv\Scripts\activate
    echo Den virtuella miljön är nu aktiverad.
) else (
    echo -----------------------------------------------
    echo Du valde att inte aktivera den virtuella miljön.
)

:: Prompta användaren för att bekräfta körning av skriptet
echo -----------------------------------------------
echo Vill du köra skriptet nu?
set /p run_choice=JA/NEJ: 

if /i "%run_choice%"=="JA" (
    echo -----------------------------------------------
    echo Kör skriptet...
    python main.py
    echo Skriptet är klart.
) else (
    echo -----------------------------------------------
    echo Du valde att inte köra skriptet.
)

:: Pausa för att hålla konsolfönstret öppet
echo -----------------------------------------------
echo Tryck på valfri tangent för att avsluta.
pause >nul

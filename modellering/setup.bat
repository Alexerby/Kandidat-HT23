@echo off

echo Vill du installera dependencies specificerade i requirements.txt?
set /p choice=JA/NEJ: 

if /i "%choice%"=="JA" (
    :: Activate the virtual environment
    call .\venv\Scripts\activate
    
    :: Install the libraries from requirements.txt
    pip install -r requirements.txt
    
    echo Installationen är klar.
    echo Tryck på valfri tangent för att avsluta.
    pause >nul
) else (
    echo Installationen avbröts.
    echo Tryck på valfri tangent för att avsluta.
    pause >nul
)

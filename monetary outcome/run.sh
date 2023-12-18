python_fil="./monetary_outcome.py"

if [ -f "$python_fil" ]; then
    python3 "$python_fil"
else
    echo "Fel: Python-filen hittades inte."
fi

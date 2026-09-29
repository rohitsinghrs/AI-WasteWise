# Run from the AI-WasteWise project root in PowerShell.
# Recommended Python version: 3.11

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python -c "import sys; print('Python:', sys.version); print('Executable:', sys.executable)"
python -c "import numpy, pandas, sklearn; print('Core ML packages: OK')"

Write-Host "`nAI WasteWise environment setup complete." -ForegroundColor Green

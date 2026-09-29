# Activate the environment first if it is not active.
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload

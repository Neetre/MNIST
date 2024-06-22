@ECHO OFF

REM SETUP THE ENVIRONMENT

python -m venv .venv

.venv/Scripts/activate

pip install -r requirements.txt

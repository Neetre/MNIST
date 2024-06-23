@ECHO OFF

CALL ./setup_Windows.bat

REM Change directory to 'bin' and run the application
CD /d bin
IF ERRORLEVEL 1 (
    ECHO Failed to change directory to 'bin'
    EXIT /B 1
)

python mnist_gui.py
IF ERRORLEVEL 1 (
    ECHO Failed to run mnist_gui.py
    EXIT /B 1
)
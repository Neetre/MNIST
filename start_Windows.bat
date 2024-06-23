@ECHO OFF

CALL ./setup/setup_Windows.bat

REM Change directory to 'bin' and run the application
CD /d bin
IF ERRORLEVEL 1 (
    ECHO Failed to change directory to 'bin'
    EXIT /B 1
)


SET /P UserInput=Want to run the GUI? (y/n):
IF /I "%UserInput%" EQU "y" GOTO RunGUI
IF /I "%UserInput%" EQU "n" GOTO End


:RunGUI
python mnist_gui.py
IF ERRORLEVEL 1 (
    ECHO Failed to run mnist_gui.py
    EXIT /B 1
)

:End
EXIT /B 0
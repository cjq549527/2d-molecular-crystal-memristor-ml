@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    set PYTHON_CMD=python
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set PYTHON_CMD=py -3
    ) else (
        echo Cannot find Python. Please install Python 3.10 or later and enable "Add Python to PATH".
        pause
        exit /b 1
    )
)

echo Using %PYTHON_CMD%
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Please check network access or install packages manually.
    pause
    exit /b 1
)

%PYTHON_CMD% run_pipeline.py
if errorlevel 1 (
    echo Pipeline failed. Please check the error message above.
    pause
    exit /b 1
)

echo.
echo Pipeline completed successfully. Results are in the results folder.
pause

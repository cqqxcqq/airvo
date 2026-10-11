@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist "requirements.txt" (
    echo requirements.txt is missing. Restore the complete project folder first.
    goto :failed
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a Python 3.12 environment...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/
        echo Include the Python launcher during installation, then try again.
        goto :failed
    )
)

".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] != (3,12))" >nul 2>&1
if errorlevel 1 (
    echo The existing environment does not run Python 3.12 correctly.
    echo Close Python programs, rename the .venv folder, and run setup again.
    goto :failed
)

echo Installing the pinned dependencies...
".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto :failed

echo Downloading and verifying the official hand and background models...
".venv\Scripts\python.exe" tools\fetch_model.py --all
if errorlevel 1 goto :failed

echo Running the automated checks...
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto :failed

echo.
echo Setup finished. Open Play-Air-Violin.cmd whenever you want to play.
echo Run this setup again after an update that changes dependencies.
pause
popd
exit /b 0

:failed
echo.
echo Setup did not finish. Read the error above before trying again.
echo Keep this window open if you need to share the error message.
pause
popd
exit /b 1

@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo Air Violin's Python environment is missing.
    echo Run Setup-Air-Violin.cmd in this folder once, then open this file again.
    goto :failed
)

".venv\Scripts\python.exe" -c "import importlib.util,sys; sys.exit(any(importlib.util.find_spec(name) is None for name in ('numpy','cv2','mediapipe','sounddevice')))" >nul 2>&1
if errorlevel 1 (
    echo Required Python packages are missing or the environment is damaged.
    echo Close Air Violin and run Setup-Air-Violin.cmd to repair it.
    goto :failed
)
".venv\Scripts\python.exe" -c "import importlib.util,sys; sys.exit(importlib.util.find_spec('pyglet') is None)" >nul 2>&1
if errorlevel 1 (
    echo 3D graphics package is missing. Run Setup-Air-Violin.cmd after updating.
    echo The automatic renderer can fall back to the 2D stage.
)

if not exist "models\hand_landmarker.task" (
    echo Default hand model is missing. Run Setup-Air-Violin.cmd for camera controls.
    echo A custom --model path or --demo can still be used.
) else if not exist "models\hand_landmarker.task.sha256" (
    echo Hand model verification file is missing. Repair it with:
    echo .venv\Scripts\python.exe tools\fetch_model.py --all --force
)
if not exist "models\selfie_segmenter_landscape.tflite" (
    echo Background model is missing. Run Setup-Air-Violin.cmd to install it.
    echo Camera video remains available without background replacement.
)

".venv\Scripts\python.exe" -m air_violin %*
set "AIR_VIOLIN_EXIT=%ERRORLEVEL%"
if not "%AIR_VIOLIN_EXIT%"=="0" (
    echo.
    echo Air Violin exited with code %AIR_VIOLIN_EXIT%. Read the message above.
    echo Try Play-Air-Violin.cmd --renderer 2d if the graphics driver cannot start 3D.
    echo Try Play-Air-Violin.cmd --demo to test sound without the camera.
    pause
)
popd
exit /b %AIR_VIOLIN_EXIT%

:failed
pause
popd
exit /b 1

@echo off
title Metro Surveillance System
color 0B

:: Activate conda environment and change directory
call conda activate metro_ai
cd proj || (
    echo Failed to change to project directory
    pause
    exit /b 1
)

:: Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:MAIN_MENU
cls
echo =================================
echo    Metro Surveillance System    
echo =================================
echo 1. Overcrowding Detection
echo 2. Fall Detection
echo 3. Unattended Object Detection
echo 4. Violence Detection
echo 5. Fire/Smoke Detection
echo 6. View Logs
echo 7. Stop All Detections
echo 8. Exit
echo =================================
echo Note: All detections run in background
echo =================================

set /p choice="Enter your choice [1-8]: "

:: Function to run a detection script in background
if "%choice%"=="1" (
    call :RUN_DETECTION "metro_surveillance.py" "Couach Crowd Density Detection"
    goto MAIN_MENU
)

if "%choice%"=="2" (
    call :RUN_DETECTION "fall_detection_final.py" "Fall Detection"
    goto MAIN_MENU
)

if "%choice%"=="3" (
    call :RUN_DETECTION "object_detection_final.py" "Object Detection"
    goto MAIN_MENU
)

if "%choice%"=="4" (
    call :RUN_DETECTION "violence_detection.py" "Violence Detection"
    goto MAIN_MENU
)

if "%choice%"=="5" (
    call :RUN_DETECTION "fire_smoke_detection_final_c.py" "Couach Crowd Density Detection"
    goto MAIN_MENU
)

if "%choice%"=="6" (
    call :VIEW_LOGS
    goto MAIN_MENU
)

if "%choice%"=="7" (
    call :STOP_DETECTIONS
    goto MAIN_MENU
)

if "%choice%"=="8" (
    call :STOP_DETECTIONS
    exit /b 0
)
echo Invalid option. Please try again.
pause
goto MAIN_MENU

:RUN_DETECTION
set script_name=%~1
set detection_name=%~2
set log_file=logs\%~n1.log

:: Check if already running
tasklist /fi "imagename eq python.exe" /fo csv | find /i "%script_name%" >nul
if %errorlevel% equ 0 (
    echo %detection_name% is already running!
    pause
    goto :EOF
)

:: Run in background and redirect output to log file
start "Metro %detection_name%" /B python "%script_name%" >> "%log_file%" 2>&1
echo %detection_name% started successfully
echo Output being logged to: %log_file%
pause
goto :EOF

:STOP_DETECTIONS
taskkill /f /im python.exe >nul 2>&1
echo All detection processes stopped
pause
goto :EOF

:VIEW_LOGS
cls
echo Available log files:
dir /B logs\*.log

:LOG_SELECT
set /p log_select="Enter log file name to view (or 'back' to return): "
if "%log_select%"=="back" goto :EOF

if not exist "logs\%log_select%" (
    echo File not found
    goto LOG_SELECT
)

more "logs\%log_select%"
pause
goto VIEW_LOGS
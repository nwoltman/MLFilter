@echo off
setlocal

set "BENCHMARK=%~dp0bin\benchmark_x64.exe"
if not exist "%BENCHMARK%" (
    echo Benchmark executable not found: "%BENCHMARK%"
    pause
    exit /b 1
)

echo MLFilter benchmark
echo.

call :run 1920 1080 nv12 || goto :failed
call :run 1920 1080 p010 || goto :failed
call :run 1280 720 nv12 || goto :failed
call :run 1280 720 p010 || goto :failed

echo.
echo Benchmarks completed successfully.
pause
exit /b 0

:run
echo.
echo ======================================================================
echo Running %1x%2 %3
echo ======================================================================
"%BENCHMARK%" --width %1 --height %2 --format %3
exit /b %errorlevel%

:failed
echo.
echo Benchmark stopped because a run failed.
pause
exit /b 1

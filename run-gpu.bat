@echo off
REM GPU-enabled startup script for Windows
REM Auto-detects best GPU configuration for the system

echo 🔍 Detecting GPU configuration...

REM Check if NVIDIA GPU is available
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ NVIDIA GPU detected

    REM Try legacy approach first (more reliable on most systems)
    echo 🐋 Trying legacy GPU configuration first...
    docker compose -f docker-compose.yml -f docker-compose.gpu-legacy.yml down 2>nul
    docker compose -f docker-compose.yml -f docker-compose.gpu-legacy.yml up --build --detach
    if %errorlevel% == 0 (
        echo ✅ Legacy GPU configuration successful
        docker compose -f docker-compose.yml -f docker-compose.gpu-legacy.yml logs -f
    ) else (
        echo ⚠️  Legacy failed, trying modern GPU syntax...
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml down 2>nul
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
    )
) else (
    echo ❌ No NVIDIA GPU detected, falling back to CPU mode
    docker compose -f docker-compose.yml -f docker-compose.cpu.yml down 2>nul
    docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
)

pause
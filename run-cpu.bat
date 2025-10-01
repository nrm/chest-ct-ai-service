@echo off
REM CPU-only startup script for Windows
REM Forces CPU mode regardless of GPU availability

echo 💻 Starting in CPU-only mode
docker compose -f docker-compose.yml -f docker-compose.cpu.yml down 2>nul
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build

pause
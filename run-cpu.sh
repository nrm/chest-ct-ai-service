#!/bin/bash
# CPU-only startup script
# Forces CPU mode regardless of GPU availability

echo "💻 Starting in CPU-only mode"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml down 2>/dev/null
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
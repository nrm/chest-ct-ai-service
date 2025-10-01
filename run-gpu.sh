#!/bin/bash
# GPU-enabled startup script
# Auto-detects best GPU configuration for the system

echo "🔍 Detecting GPU configuration..."

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"

    # Try legacy approach first (more reliable on most systems)
    echo "🐋 Trying legacy GPU configuration first..."
    docker compose -f docker-compose.yml -f docker-compose.gpu-legacy.yml down 2>/dev/null
    if docker compose -f docker-compose.yml -f docker-compose.gpu-legacy.yml up --build --detach; then
        echo "✅ Legacy GPU configuration successful"
        docker compose -f docker-compose.yml -f docker-compose.gpu-legacy.yml logs -f
    else
        echo "⚠️  Legacy failed, trying modern GPU syntax..."
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml down 2>/dev/null
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
    fi
else
    echo "❌ No NVIDIA GPU detected, falling back to CPU mode"
    docker compose -f docker-compose.yml -f docker-compose.cpu.yml down 2>/dev/null
    docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
fi
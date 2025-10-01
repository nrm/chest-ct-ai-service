# RadiAssist API Build Guide

## 🚀 Quick Start

> 📖 **Для пользователей**: См. [DEPLOYMENT.md](DEPLOYMENT.md) для полного руководства по развертыванию с GPU/CPU поддержкой

### Production Build (Recommended)
```bash
# Build with optimized caching
docker compose build

# Run API service
docker compose up -d

# Check logs
docker compose logs -f
```

### Development Build (Hot Reload)
```bash
# Uncomment source volume mounts in docker-compose.yml
# Then build and run
docker compose up --build
```

## 📦 Build Stages Optimization

### Stage 1: Dependencies (~3-5 minutes first time, cached afterwards)
- System packages (gcc, g++, curl)
- Python packages (torch, fastapi, etc.)
- **Cached unless `requirements.txt` changes**

### Stage 2: Application Code (~10 seconds)
- Python source files
- **Rebuilds when `.py` files change**

### Stage 3: Model Weights (~30 seconds)
- ML model weights (122MB via Git LFS)
- **Cached unless model files change**

### Stage 4: Runtime (~5 seconds)
- Directory setup and configuration
- **Always rebuilds (minimal)**

## 🎯 Cache Efficiency

### Full rebuild (first time): ~5-8 minutes
### Code changes only: ~15 seconds
### Requirements changes: ~3-5 minutes
### Model changes: ~30 seconds

## 🔧 Build Arguments

```bash
# Custom Python version
docker compose build --build-arg PYTHON_VERSION=3.12

# Target specific stage
docker compose build --target application  # Skip models for testing
```

## 📁 File Structure Impact

### High cache hit (rarely change):
- `requirements.txt` → Dependencies stage
- `models/*.pth` → Model weights stage

### Medium cache hit:
- `main.py`, `hackathon/`, `radiassist/` → Application stage

### Always rebuild:
- Runtime configuration

## 🐛 Troubleshooting

### Clear all cache:
```bash
docker compose build --no-cache
```

### Clear specific stage:
```bash
docker compose build --no-cache --target dependencies
```

### Check layer sizes:
```bash
docker history radiassist-api:latest
```

## 💡 Best Practices

1. **Change `requirements.txt` sparingly** - triggers long rebuild
2. **Group related code changes** - minimizes rebuilds
3. **Use `.dockerignore`** - excludes unnecessary files
4. **Enable BuildKit** - better caching and parallelization
5. **Monitor layer sizes** - keep images lean
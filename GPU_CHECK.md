# 🔍 Проверка GPU в RadiAssist

## Быстрые способы проверки

### 1. **Проверка через API (если контейнер запущен)**
```bash
# Проверка статуса GPU
curl -s http://localhost:8000/gpu-status | python3 -m json.tool

# Проверка здоровья сервиса
curl -s http://localhost:8000/health
```

### 2. **Проверка через скрипт диагностики**
```bash
# Запуск диагностического скрипта
python3 check_gpu.py
```

### 3. **Проверка логов запуска**
```bash
# Просмотр логов с GPU диагностикой
docker compose -f docker-compose.new.yml logs backend | grep -E "(GPU|nvidia|CUDA|PyTorch)"

# Просмотр всех логов запуска
docker compose -f docker-compose.new.yml logs backend | head -100
```

### 4. **Проверка Docker GPU проброса**
```bash
# Проверка, что контейнер видит GPU
docker exec -it $(docker ps -q --filter "name=backend") nvidia-smi

# Проверка переменных окружения в контейнере
docker exec -it $(docker ps -q --filter "name=backend") env | grep -E "(CUDA|NVIDIA)"
```

## 🚀 Запуск с GPU

### Способ 1: Через run-gpu.sh
```bash
./run-gpu.sh
```

### Способ 2: Через docker-compose с GPU
```bash
# Остановить текущий контейнер
docker compose -f docker-compose.new.yml down

# Запустить с GPU поддержкой
docker compose -f docker-compose.new.yml -f docker-compose.gpu.yml up --build -d

# Или с legacy GPU поддержкой
docker compose -f docker-compose.new.yml -f docker-compose.gpu-legacy.yml up --build -d
```

### Способ 3: Прямой запуск с GPU
```bash
# Остановить текущий контейнер
docker compose -f docker-compose.new.yml down

# Запустить с GPU
docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs radiassist-api:latest
```

## 🔧 Диагностика проблем

### Если GPU не работает:

1. **Проверьте NVIDIA драйверы:**
   ```bash
   nvidia-smi
   ```

2. **Проверьте nvidia-docker2:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. **Проверьте Docker Compose версию:**
   ```bash
   docker compose version
   ```

4. **Проверьте права доступа:**
   ```bash
   ls -la /dev/nvidia*
   ```

### Ожидаемый вывод при успешной работе GPU:

```
🔍 GPU Diagnostics at startup:
✅ nvidia-smi command successful:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 3080    Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   42C    P8    15W / 320W |      0MiB / 10240MiB |      0%      Default |
|                               |                      |                  N/A |
+-----------------------------------------------------------------------------+

🔥 PyTorch CUDA available: True
🔥 PyTorch CUDA device count: 1
🔥 PyTorch current device: 0
🔥 PyTorch device name: GeForce RTX 3080
```

## ⚡ Производительность

### С GPU:
- Время обработки: 10-60 секунд
- Использование GPU: 80-95%
- Память GPU: 2-8 GB

### Без GPU (CPU only):
- Время обработки: 5-30 минут
- Использование CPU: 100%
- Память RAM: 4-16 GB

## 🆘 Если ничего не помогает

1. Перезагрузите систему
2. Обновите NVIDIA драйверы
3. Переустановите nvidia-docker2
4. Проверьте совместимость версий CUDA и PyTorch
5. Запустите с `--privileged` флагом

```bash
docker run --privileged --gpus all -p 8000:8000 radiassist-api:latest
```

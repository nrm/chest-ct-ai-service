# 🚀 Руководство по развертыванию RadiAssist API

## Быстрый запуск

> 💡 **Для разработчиков**: См. [BUILD.md](BUILD.md) для информации об оптимизации сборки Docker образов

### Вариант 1: Автоопределение (Рекомендуется)

**Linux/macOS:**
```bash
./run-gpu.sh
```

**Windows:**
```cmd
run-gpu.bat
```

Автоматически определяет доступность GPU и использует оптимальную конфигурацию.

### Вариант 2: Ручная настройка

#### Режим GPU (современный Docker Compose)
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

#### Режим GPU (устаревшие системы)
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu-legacy.yml up --build
```

#### Режим только CPU
**Linux/macOS:**
```bash
./run-cpu.sh
```

**Windows:**
```cmd
run-cpu.bat
```

**Или вручную:**
```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

## Системные требования

### Операционные системы
- ✅ **Ubuntu 22.04 LTS** (полностью протестировано)
- 🔄 **Windows 10/11** с WSL2 (в процессе тестирования)
- ⚠️ **Другие Linux** дистрибутивы (должны работать, но не тестировались)

### Требования для GPU режима
- **NVIDIA GPU** с поддержкой CUDA
- **NVIDIA драйверы** (450.80.02+)
- **nvidia-container-toolkit** установлен
- **Docker** с GPU поддержкой

### Резервный режим CPU
- **8GB+ RAM** рекомендуется (протестировано на 16GB+)
- **Многоядерный процессор** для приемлемой производительности
- **Docker** без дополнительных требований

## Предварительная установка

### Ubuntu/Debian

#### 1. Обновление системы
```bash
sudo apt update && sudo apt upgrade -y
```

#### 2. Установка базовых зависимостей
```bash
# Базовые утилиты
sudo apt install -y curl wget gnupg software-properties-common

# Проверка поддержки NVIDIA GPU (опционально)
lspci | grep -i nvidia
```

#### 3. Установка NVIDIA драйверов (для GPU режима)
```bash
# Автоматическая установка рекомендуемых драйверов
sudo ubuntu-drivers autoinstall

# ИЛИ ручная установка конкретной версии
sudo apt install -y nvidia-driver-470 nvidia-dkms-470

# Перезагрузка после установки драйверов
sudo reboot

# Проверка после перезагрузки
nvidia-smi
```

#### 4. Установка Docker
```bash
# Удаление старых версий
sudo apt remove docker docker-engine docker.io containerd runc

# Установка Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Добавление пользователя в группу docker
sudo usermod -aG docker $USER

# Запуск Docker
sudo systemctl enable docker
sudo systemctl start docker

# Выход и повторный вход для применения группы
newgrp docker
```

#### 5. Установка NVIDIA Container Toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
sudo systemctl restart docker
```

#### 6. Проверка установки
```bash
# Проверка Docker
docker --version
docker run hello-world

# Проверка NVIDIA в Docker (только для GPU)
docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```


### Windows

#### 1. Системные требования
- **Windows 10** версия 2004+ (build 19041+) или **Windows 11**
- **WSL2** поддержка
- **NVIDIA GPU** с архитектурой Kepler или новее
- **16GB+ RAM** рекомендуется

#### 2. Включение WSL2
```powershell
# Запустить PowerShell от имени администратора
# Включить WSL и Virtual Machine Platform
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Перезагрузка
shutdown /r /t 0

# После перезагрузки - установить WSL2 как по умолчанию
wsl --set-default-version 2

# Установить Ubuntu в WSL2
wsl --install -d Ubuntu-20.04
```

#### 3. Установка NVIDIA драйверов для Windows
```powershell
# Скачайте и установите последние NVIDIA драйверы (версия 451.48+)
# с официального сайта: https://www.nvidia.com/drivers

# ИЛИ используйте GeForce Experience для автоматического обновления

# Проверка после установки
nvidia-smi
```

#### 4. Установка Docker Desktop
1. **Скачайте Docker Desktop** с официального сайта
2. **Установите с WSL2 backend**
3. **Включите WSL2 интеграцию** в настройках Docker Desktop
4. **Включите GPU support** в Docker Desktop settings

#### 5. Настройка WSL2 GPU поддержки
```powershell
# Проверьте GPU в WSL2
wsl nvidia-smi

# Если не работает - обновите WSL2
wsl --update
wsl --shutdown

# Перезапустите WSL
wsl -d Ubuntu-20.04
```

#### 6. Установка зависимостей в WSL2
```bash
# Внутри WSL2 Ubuntu
sudo apt update
sudo apt install -y curl wget git

# Проверка Docker в WSL2
docker --version
docker run hello-world

# Проверка GPU поддержки
docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## Протестированные платформы

- ✅ **Ubuntu 22.04 LTS** - полностью протестировано и работает с GPU
- 🔄 **Windows 10/11** - планируется тестирование с WSL2 + Docker Desktop

**Примечание**: Другие Linux дистрибутивы (CentOS, RHEL, Debian) должны работать аналогично, но не тестировались. macOS поддерживает только CPU режим (без NVIDIA GPU).

## Проверка готовности системы

### Полная проверка для GPU режима
```bash
# 1. Проверка NVIDIA драйвера
nvidia-smi

# 2. Проверка Docker
docker --version
docker run hello-world

# 3. Проверка NVIDIA Container Toolkit
docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# 4. Проверка Docker Compose
docker compose version

# 5. Если все работает - можно запускать RadiAssist
./run-gpu.sh
```

### Минимальная проверка для CPU режима
```bash
# 1. Проверка Docker
docker --version
docker run hello-world

# 2. Проверка Docker Compose
docker compose version

# 3. Запуск в CPU режиме
./run-cpu.sh
```

## Структура файлов
```
api/
├── docker-compose.yml              # Базовая конфигурация
├── docker-compose.gpu.yml          # Современное GPU расширение
├── docker-compose.gpu-legacy.yml   # Устаревшее GPU расширение
├── docker-compose.cpu.yml          # CPU-only расширение
├── run-gpu.sh / run-gpu.bat        # Скрипт автоопределения
├── run-cpu.sh / run-cpu.bat        # Скрипт CPU-only
└── DEPLOYMENT.md                   # Этот файл
```

## Устранение неполадок

### GPU не определяется
```bash
# Проверьте GPU
nvidia-smi

# Проверьте поддержку GPU в Docker
docker run --gpus all --rm ubuntu nvidia-smi

# Перезапустите Docker daemon
sudo systemctl restart docker
```

**Windows:**
```powershell
# Проверьте GPU в WSL2
wsl nvidia-smi

# Проверьте Docker Desktop GPU
docker run --gpus all --rm ubuntu nvidia-smi
```

### Проблемы с правами доступа
**Linux:**
```bash
# Добавьте пользователя в группу docker
sudo usermod -aG docker $USER
# Выйдите и войдите обратно

# Исправьте права устройств (при необходимости)
sudo chmod 666 /dev/nvidia*
```

**Windows:**
- Запустите PowerShell/CMD от имени администратора
- Убедитесь, что Docker Desktop запущен с правами администратора

### Проблемы с памятью (режим CPU)
- Уменьшите `MAX_CONCURRENT_JOBS` до 1
- Увеличьте файл подкачки системы
- Используйте меньшие варианты моделей

## Параметры конфигурации

### Переменные окружения
- `MAX_CONCURRENT_JOBS=2` - Количество параллельных задач
- `PROCESSING_TIMEOUT=600` - Таймаут на задачу (секунды)
- `CUDA_VISIBLE_DEVICES=0` - Выбор GPU устройства
- `FORCE_CPU_MODE=true` - Принудительный режим CPU

### Ограничения ресурсов
- **Режим GPU**: ограничение 8GB RAM
- **Режим CPU**: ограничение 4GB RAM
- **Таймаут обработки**: 10 минут на исследование

## Ожидаемая производительность

| Режим | Время обработки/исследование | Использование памяти | Энергопотребление |
|-------|------------------------------|---------------------|-------------------|
| GPU (A30) | ~3-5 секунд | 2-4GB | 100-150W |
| CPU (16 ядер) | ~2-5 минут | 4-8GB | 50-100W |
| CPU (Windows) | ~5-10 минут | 6-12GB | 80-150W |

## Production развертывание

1. **Используйте конкретные теги образов** вместо `:latest`
2. **Настройте ротацию логов** (настроено в compose файлах)
3. **Настройте reverse proxy** (nginx/traefik)
4. **Настройте мониторинг** (Prometheus/Grafana)
5. **Регулярные бэкапы** тома `/app/data`

## Развертывание на нескольких хостах

### Docker Swarm
```bash
# GPU узлы
docker node update --label-add gpu=nvidia NODE_NAME

# Развертывание с ограничениями
docker stack deploy -c docker-compose.yml -c docker-compose.gpu.yml radiassist
```

### Kubernetes
Используйте NVIDIA Device Plugin и селекторы GPU узлов:
```yaml
nodeSelector:
  accelerator: nvidia-tesla-a30
resources:
  limits:
    nvidia.com/gpu: 1
```

## Особенности Windows

### WSL2 Backend (Рекомендуется)
- Лучшая производительность
- Полная поддержка GPU
- Нативная поддержка Linux контейнеров

### Hyper-V Backend (Устаревший)
- Только CPU режим
- Медленнее производительность
- Ограниченная функциональность

### Команды PowerShell
```powershell
# Запуск с GPU
.\run-gpu.bat

# Запуск только CPU
.\run-cpu.bat

# Проверка статуса
docker ps
docker logs radiassist-api
```
#!/bin/bash

# RadiAssist Startup Script
# Запуск полного стека RadiAssist

echo "🚀 Запуск RadiAssist - Анализ КТ грудной клетки"
echo "=============================================="
echo "🔥 GPU поддержка включена по умолчанию"

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker и попробуйте снова."
    exit 1
fi

# Проверка наличия Docker Compose
if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose не установлен или не доступен."
    exit 1
fi

echo "✅ Docker и Docker Compose обнаружены"

# Создание необходимых директорий
echo "📁 Создание директорий данных..."
mkdir -p ./data
mkdir -p ./logs
mkdir -p ./frontend/dist

# Проверка наличия моделей ИИ
if [ ! -f "./models/covid19_classifier_fold1_best_auc.pth" ]; then
    echo "⚠️  Модели ИИ не найдены. Убедитесь, что Git LFS настроен правильно:"
    echo "   git lfs install"
    echo "   git lfs pull"
fi

# Определение режима запуска
MODE=${1:-development}

echo "🔧 Режим запуска: $MODE"

case $MODE in
    "production")
        echo "🏭 Запуск в production режиме (с nginx reverse proxy и GPU)..."
        docker compose -f docker-compose.new.yml --profile production up --build -d
        ;;
    "development")
        echo "🔧 Запуск в development режиме с GPU поддержкой..."
        docker compose -f docker-compose.new.yml -f docker-compose.new.gpu-legacy.yml up --build -d
        ;;
    "remote")
        echo "🌐 Запуск frontend с подключением к удаленному backend..."
        echo "   Backend URL: ${RADIASSIST_BACKEND_URL:-http://91.151.182.59:8000}"
        docker compose -f docker-compose.new.yml --profile remote up --build -d
        ;;
    "remote-nginx")
        echo "🌐 Production режим с удаленным backend (nginx на порту 8080)..."
        echo "   Backend URL: ${RADIASSIST_BACKEND_URL:-http://91.151.182.59:8000}"
        docker compose -f docker-compose.new.yml --profile remote-nginx up --build -d
        ;;
    "backend-only")
        echo "🔧 Запуск только backend API с GPU..."
        docker compose -f docker-compose.new.yml up --build -d
        ;;
    "frontend-only")
        echo "🎨 Запуск только frontend с подключением к backend:8000..."
        docker compose -f docker-compose.new.yml --profile frontend-only up --build -d
        ;;
    "cpu-only")
        echo "🖥️  Запуск в CPU режиме (без GPU)..."
        # Создаем временный файл без GPU настроек
        cp docker-compose.new.yml docker-compose.cpu-temp.yml
        sed -i '/privileged: true/d; /runtime: nvidia/d; /devices:/,/dev\/nvidia-uvm-tools/d; /\/dev\/nvidia-caps/d' docker-compose.cpu-temp.yml
        docker compose -f docker-compose.cpu-temp.yml up --build -d
        rm docker-compose.cpu-temp.yml
        ;;
    "stop")
        echo "🛑 Остановка всех сервисов..."
        docker compose -f docker-compose.new.yml down
        exit 0
        ;;
    "logs")
        echo "📋 Просмотр логов..."
        docker compose -f docker-compose.new.yml logs -f
        exit 0
        ;;
    "gpu-check")
        echo "🔍 Проверка GPU..."
        python3 check_gpu.py
        exit 0
        ;;
    "gpu-status")
        echo "🔍 Статус GPU через API..."
        curl -s http://localhost:8000/gpu-status | python3 -m json.tool 2>/dev/null || echo "❌ API недоступен. Запустите сервис сначала."
        exit 0
        ;;
    *)
        echo "❌ Неизвестный режим: $MODE"
        echo ""
        echo "Доступные режимы:"
        echo "  development     - Локальный backend + frontend с GPU (по умолчанию)"
        echo "  production      - Production с nginx reverse proxy и GPU"
        echo "  cpu-only        - Запуск без GPU (только CPU)"
        echo "  remote          - Только frontend, подключается к удаленному backend"
        echo "  remote-nginx    - Nginx + frontend, подключается к удаленному backend"
        echo "  backend-only    - Только backend API с GPU"
        echo "  frontend-only   - Только frontend, подключается к backend:8000"
        echo "  stop           - Остановить все сервисы"
        echo "  logs           - Просмотр логов"
        echo "  gpu-check      - Проверка GPU (локальная диагностика)"
        echo "  gpu-status     - Статус GPU через API"
        echo ""
        echo "Переменные окружения:"
        echo "  RADIASSIST_BACKEND_URL - URL удаленного backend (по умолчанию: http://91.151.182.59:8000)"
        echo "  CUDA_VISIBLE_DEVICES   - GPU устройства"
        echo ""
        echo "Примеры:"
        echo "  $0                                           # Development с GPU (по умолчанию)"
        echo "  $0 cpu-only                                  # Запуск без GPU"
        echo "  $0 remote                                    # Frontend на порту 3001"
        echo "  $0 remote-nginx                             # Nginx на порту 8080"
        echo "  RADIASSIST_BACKEND_URL=http://my-api.com:8000 $0 remote"
        exit 1
        ;;
esac

# Ожидание запуска сервисов
echo "⏳ Ожидание запуска сервисов..."
sleep 10

# Проверка здоровья сервисов
echo "🔍 Проверка здоровья сервисов..."

# Backend health check
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ Backend API: http://localhost:8000"
    echo "   📖 Документация API: http://localhost:8000/docs"
else
    echo "⚠️  Backend API недоступен на http://localhost:8000"
fi

# Frontend health check
if curl -f http://localhost:3000 &> /dev/null; then
    echo "✅ Frontend: http://localhost:3000"
else
    echo "⚠️  Frontend недоступен на http://localhost:3000"
fi

# Remote frontend health check
if [ "$MODE" = "remote" ]; then
    if curl -f http://localhost:3001 &> /dev/null; then
        echo "✅ Remote Frontend: http://localhost:3001"
    else
        echo "⚠️  Remote Frontend недоступен на http://localhost:3001"
    fi
fi

# Production health check (nginx)
if [ "$MODE" = "production" ]; then
    if curl -f http://localhost &> /dev/null; then
        echo "✅ Production: http://localhost"
    else
        echo "⚠️  Production недоступен на http://localhost"
    fi
fi

# Remote nginx health check
if [ "$MODE" = "remote-nginx" ]; then
    if curl -f http://localhost:8080 &> /dev/null; then
        echo "✅ Remote Nginx: http://localhost:8080"
    else
        echo "⚠️  Remote Nginx недоступен на http://localhost:8080"
    fi
fi

echo ""
echo "🎉 RadiAssist запущен!"
echo ""
echo "Доступ к приложению:"
case $MODE in
    "production")
        echo "   🌐 http://localhost (nginx reverse proxy)"
        echo "   🔗 http://localhost/api/ (backend API)"
        echo "   📖 http://localhost/docs (документация API)"
        ;;
    "remote")
        echo "   🌐 http://localhost:3001 (frontend)"
        echo "   🔗 Backend: ${RADIASSIST_BACKEND_URL:-http://91.151.182.59:8000}"
        ;;
    "remote-nginx")
        echo "   🌐 http://localhost:8080 (nginx)"
        echo "   🔗 Backend: ${RADIASSIST_BACKEND_URL:-http://91.151.182.59:8000}"
        ;;
    "development"|"full")
        echo "   🌐 http://localhost:3000 (frontend)"
        echo "   🔗 http://localhost:8000 (backend API)"
        echo "   📖 http://localhost:8000/docs (документация API)"
        ;;
    "backend-only")
        echo "   🔗 http://localhost:8000 (backend API)"
        echo "   📖 http://localhost:8000/docs (документация API)"
        ;;
    "frontend-only")
        echo "   🌐 http://localhost:3000 (frontend)"
        echo "   🔗 http://localhost:8000 (backend API)"
        ;;
esac
echo ""
echo "Управление:"
echo "   Остановить: $0 stop"
echo "   Логи: $0 logs"
echo ""
echo "=============================================="

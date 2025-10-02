# Изменения конфигурации

## Реструктуризация проекта (2024-10-02)

### Что изменилось:

1. **Перенос frontend внутрь chest-ct-ai-service**
   - ✅ Frontend перемещен из `./frontend/` в `./chest-ct-ai-service/frontend/`
   - ✅ Переименование `chest-ct-ai-service-dev` обратно в `chest-ct-ai-service`
   - ✅ Обновлены все пути в `docker-compose.yml`

2. **Настройка GPU с privileged режимом**
   - ✅ Добавлен `privileged: true` в `docker-compose.gpu.yml`
   - ✅ Добавлен `privileged: true` в `docker-compose.gpu-legacy.yml`
   - ✅ Теперь GPU запуск работает с необходимыми привилегиями

3. **Добавление возможности изменения лимита времени обработки**
   - ✅ Добавлен параметр `processing_timeout` в эндпоинт `/upload` (по умолчанию 600 секунд)
   - ✅ Обновлены функции обработки для поддержки настраиваемого timeout
   - ✅ Создана модель `UploadRequest` для будущего расширения API

4. **Перемещение дополнительных утилит**
   - ✅ Папка `segment_and_viz_1` перемещена в `chest-ct-ai-service/segment_and_viz_1`

### Обновленные файлы:
- `docker-compose.new.yml` - новый файл с исправленными путями к сервисам
- `docker-compose.yml` - восстановлен оригинальный файл через git restore
- `docker-compose.gpu.yml` - добавлен privileged режим
- `docker-compose.gpu-legacy.yml` - добавлен privileged режим
- `main.py` - добавлена поддержка настраиваемого timeout
- `start.sh` - обновлен для использования -f docker-compose.new.yml
- `README.md` - обновлена структура проекта

### Новая структура проекта:
```
chest-ct-ai-service/              # Полный самодостаточный проект
├── frontend/                     # React приложение
├── segment_and_viz_1/            # Дополнительные утилиты
├── docker-compose.gpu.yml        # GPU конфигурация с privileged
├── docker-compose.yml            # Оригинальная оркестрация (сохранена)
├── docker-compose.new.yml        # Новая оркестрация с frontend
├── nginx.conf                    # Nginx конфигурация
├── start.sh                      # Скрипт запуска (использует .new.yml)
└── main.py                       # API с поддержкой timeout
```

### Использование нового API:
```bash
# Загрузка с настройкой timeout (по умолчанию 600 секунд)
curl -X POST "http://localhost:8000/upload?processing_timeout=900" \
  -F "file=@your_dicom.zip"
```

### Запуск проекта:
```bash
# Перейти в папку проекта
cd chest-ct-ai-service

# Запуск в development режиме
./start.sh development

# Запуск на GPU
./run-gpu.sh

# Запуск в production режиме
./start.sh production
```

---

## Удален хардкод URL удаленного сервера

### Что изменилось:

1. **frontend/src/services/api.ts**
   - ❌ Было: `return 'http://91.151.182.59:8000';`
   - ✅ Стало: `return 'http://localhost:8000';`
   - По умолчанию фронтенд теперь обращается к локальному бэкенду

2. **frontend/nginx.conf**
   - ❌ Было: `proxy_pass http://91.151.182.59:8000/;`
   - ✅ Стало: `proxy_pass http://backend:8000/;`
   - Nginx теперь использует внутреннее имя сервиса Docker

3. **docker-compose.yml**
   - Удалены дефолтные значения с хардкодом из переменных окружения
   - Теперь `RADIASSIST_BACKEND_URL` должна быть явно указана для remote режимов

4. **chest-ct-ai-service-dev/main.py**
   - ✅ Добавлен CORS middleware
   - Теперь API принимает запросы с любых origins (для development)

### Режимы работы:

#### Development (по умолчанию)
```bash
./start.sh
# или
./start.sh development
```
- Фронтенд на http://localhost:3000
- Бэкенд на http://localhost:8000
- Фронтенд обращается к локальному бэкенду

#### Production
```bash
./start.sh production
```
- Все через nginx на http://localhost
- Фронтенд и бэкенд работают локально

#### Remote (подключение к удаленному серверу)
```bash
RADIASSIST_BACKEND_URL=http://your-server:8000 ./start.sh remote
```
- Фронтенд на http://localhost:3001
- Подключается к указанному удаленному бэкенду
- Дефолтное значение в start.sh: http://91.151.182.59:8000

### Важно:

✅ Хардкод URL остался только в **start.sh** как дефолтное значение для переменной `RADIASSIST_BACKEND_URL`
✅ Это единственное место где указан удаленный сервер - для удобства быстрого старта в remote режиме
✅ При запуске просто `./start.sh` все работает локально

### CORS настроен:

Backend теперь принимает запросы от любых источников благодаря CORSMiddleware.
Для production рекомендуется ограничить список origins.


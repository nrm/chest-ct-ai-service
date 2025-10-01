# RadiAssist API Сервис

Простой REST API для анализа КТ грудной клетки с использованием пайплайна RadiAssist хакатона.

## Предварительные требования

### 1. Клонирование репозитория с весами моделей

Проект использует **Git LFS** для хранения больших файлов моделей (~400МБ). Необходимо установить Git LFS и загрузить веса моделей перед сборкой:

```bash
# Установка Git LFS (если еще не установлен)
# Ubuntu/Debian:
sudo apt install git-lfs

# Windows:
# Скачать с https://git-lfs.github.io/

# Инициализация Git LFS
git lfs install

# Клонирование репозитория с LFS файлами
git clone <repository-url>
cd radiassist-chest/api

# Проверка загрузки весов моделей
ls -lh models/
# Должно показать:
# covid19_triage_mlflow_20250925_110817.pth (~65МБ)
# luna16_detector_20250923_191335_best.pth (~57МБ)
```

### 2. Настройка системы

Для детальной настройки системы (Docker, GPU драйверы и т.д.) см. **[DEPLOYMENT.md](DEPLOYMENT.md)**.

## Быстрый запуск

### Использование Docker (Рекомендуется)

```bash
# Режим GPU (автоопределение)
./run-gpu.sh

# Режим CPU
./run-cpu.sh

# Ручной Docker Compose
docker-compose up --build

# API будет доступен по адресу http://localhost:8000
```

**Примечание**: Используйте `./run-gpu.sh` для автоматического определения GPU/CPU и оптимальной производительности.

### Локальная разработка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
python main.py
```

## Использование API

### 1. Загрузка DICOM ZIP файла

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_dicom_study.zip"
```

Ответ:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "File uploaded successfully. Found 451 DICOM files."
}
```

### 2. Проверка статуса задачи

```bash
curl "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/status"
```

Ответ:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2025-09-29T10:00:00",
  "completed_at": "2025-09-29T10:03:45",
  "result_files": {
    "excel": "/tmp/radiassist_result.xlsx",
    "csv": "/tmp/radiassist_result.csv"
  }
}
```

### 3. Скачивание результатов

```bash
# Скачать Excel файл с оригинальным именем
curl -J -O "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/result/excel"
# Создает: norma_anon_result.xlsx

# Скачать CSV файл с оригинальным именем
curl -J -O "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/result/csv"
# Создает: norma_anon_result.csv

# Альтернатива: указать имя файла вручную
curl -o "my_custom_name.csv" "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/result/csv"
```

**Примечание**: Флаг `-J` указывает curl использовать имя файла из заголовка Content-Disposition сервера, что сохраняет оригинальное имя ZIP файла в файлах результатов.

### 4. Просмотр сохраненных данных

```bash
# Список всех файлов задачи
curl "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/files"

# Просмотр директории данных
curl "http://localhost:8000/data/browse"
```

## Эндпоинты API

- `POST /upload` - Загрузка DICOM ZIP файла
- `GET /tasks/{task_id}/status` - Получение статуса задачи
- `GET /tasks/{task_id}/result/excel` - Скачивание Excel результата
- `GET /tasks/{task_id}/result/csv` - Скачивание CSV результата
- `GET /tasks/{task_id}/files` - Список файлов задачи
- `GET /tasks` - Список всех задач
- `GET /data/browse` - Просмотр директории данных
- `GET /health` - Проверка работоспособности

## Значения статусов задач

- `pending` - Задача создана, ожидает запуска
- `processing` - В данный момент анализируются DICOM данные
- `completed` - Анализ завершен успешно
- `failed` - Произошла ошибка во время обработки

## Формат вывода

Результаты соответствуют требованиям хакатона:

**Колонки Excel/CSV:**
- `path_to_study` - Оригинальное имя ZIP файла
- `study_uid` - DICOM Study UID
- `series_uid` - DICOM Series UID
- `probability_of_pathology` - Вероятность патологии (0.0-1.0)
- `pathology` - Бинарная классификация (0=норма, 1=патология)
- `processing_status` - Успех/Неудача
- `time_of_processing` - Время обработки в секундах
- `most_dangerous_pathology_type` - Тип обнаруженной патологии
- `pathology_localization` - Координаты (x_min,x_max,y_min,y_max,z_min,z_max)

## Хранение данных

Файлы постоянно хранятся в следующей структуре:

```
data/
├── uploads/
│   ├── task_uuid_1/
│   │   └── norma_anon.zip
│   └── task_uuid_2/
│       └── pneumonia_anon.zip
└── results/
    ├── task_uuid_1/
    │   ├── hackathon_test_results.xlsx
    │   └── hackathon_test_results.csv
    └── task_uuid_2/
        ├── hackathon_test_results.xlsx
        └── hackathon_test_results.csv
```

Это позволяет:
- **Исследование данных**: Загруженные ZIP файлы сохранены для анализа
- **Сохранение результатов**: Excel/CSV файлы остаются доступными
- **Простая отладка**: Полная история данных для устранения неполадок

## Производительность и ограничения

- Время обработки: 3-60 секунд на исследование (в зависимости от размера)
- Использование памяти: ~2-8ГБ во время обработки
- Поддержка DICOM ZIP файлов до 1ГБ
- **Параллельные задачи**: Максимум 2 одновременные задачи обработки (настраивается)
- **Таймаут**: 10 минут на задачу (600 секунд, настраивается)
- **Лимит очереди**: Возвращает HTTP 429 если очередь обработки переполнена

## Конфигурация

Установите переменные окружения для настройки ограничений:

```bash
# Максимальное количество параллельных задач обработки
export MAX_CONCURRENT_JOBS=3

# Таймаут обработки в секундах
export PROCESSING_TIMEOUT=900

# Запуск API
python main.py
```

Или в docker-compose.yml:
```yaml
environment:
  - MAX_CONCURRENT_JOBS=3
  - PROCESSING_TIMEOUT=900
```

## Устранение неполадок

### Проблемы с Git LFS

Если веса моделей отсутствуют или повреждены:

```bash
# Проверка статуса Git LFS
git lfs status

# Повторная загрузка LFS файлов
git lfs pull

# Проверка размеров файлов моделей
ls -lh models/
# covid19_triage_mlflow_20250925_110817.pth должен быть ~65МБ
# luna16_detector_20250923_191335_best.pth должен быть ~57МБ

# Если файлы маленькие (< 1КБ), это LFS указатели - скачайте их:
git lfs checkout
```

### Частые ошибки Git LFS

**Ошибка**: `git-lfs smudge -- 'models/...': git-lfs: command not found`
```bash
# Установите Git LFS и повторите
git lfs install
git lfs pull
```

**Ошибка**: Файлы показывают текстовые указатели вместо бинарных данных
```bash
# Это означает, что LFS файлы не были загружены
git lfs pull
# или
git lfs checkout
```

### Проблемы сборки Docker

**Ошибка**: `ModuleNotFoundError` или `FileNotFoundError` для моделей
```bash
# Убедитесь, что модели присутствуют перед сборкой
ls -lh models/
# Если отсутствуют, повторно загрузите LFS файлы
git lfs pull
```

## Обработка ошибок

API предоставляет детализированные сообщения об ошибках для:
- Неверные форматы файлов [TODO]
- Поврежденные ZIP файлы [TODO]
- Отсутствующие DICOM файлы [TODO]
- Ошибки обработки [TODO]
- Неверная анатомия/модальность [TODO]
- **Отсутствующие веса моделей** (проверьте настройку Git LFS)
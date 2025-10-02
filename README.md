# RadiAssist API Сервис

REST API для анализа КТ грудной клетки с использованием AI моделей для автоматической классификации "норма/патология".

## 🎯 Архитектура системы

### Обзор pipeline

Система использует **гибридный подход**, комбинируя глубокое обучение (CNN) с медицинскими правилами (KSL):

```
DICOM Input → Validation → COVID19 CNN → KSL Analysis → Hybrid Aggregation → Final Decision
```

### Компоненты системы

#### 1️⃣ COVID19 Classifier (ResNet50 2D MIL)
**Назначение:** Обнаружение паттернов патологии на КТ грудной клетки
**Архитектура:**
- ResNet50 энкодер (предобученный на ImageNet)
- Attention-based Multiple Instance Learning (MIL)
- Обработка 64 осевых срезов, разрешение 256×256
- Выход: вероятность патологии [0.0-1.0]

**Особенности:**
- Обучен на 1,210 исследованиях (COVID19_1110 + MosMedData Cancer)
- Val AUC: **0.9711** (лучший фолд: 0.9839)
- Размер модели: 294MB

#### 2️⃣ KSL Z-profile Analyzer (ct_z.py)
**Назначение:** Скрининг по срезам - извлечение лёгочных признаков и построение профилей по оси Z

**Алгоритм:**
1. Пер-срезная сегментация лёгких (порог HU < -500 + морфология)
2. Разделение на левое/правое лёгкое
3. Извлечение признаков на каждом срезе:
   - **Площадь лёгких** (lung_area_px)
   - **Средний HU** (mean_lung_HU) - общая аэрация
   - **Плотность >-500 HU** (frac_dense_m500) - консолидация, инфильтраты
   - **Плотность >-300 HU** (frac_dense_m300) - выраженная консолидация
   - **Эмфизема <-950 HU** (frac_emph_m950) - воздушные ловушки
   - **L/R асимметрия** (LR_asym) - односторонние поражения

**Скор аномальности [0.0-1.0]:**
```
Z-score = 0.45×dense_m500 + 0.25×mean_HU + 0.10×LR_asym + 0.10×(1-lung_area)
```

**Выход:**
- Z-profile score - композитный индекс аномальности
- Медицинские признаки для интерпретации
- Подозрительные z-участки (score > 0.7)

#### 3️⃣ Hybrid Decision Logic (адаптивная калибровка)
**Назначение:** Интеграция CNN + KSL для финального решения
**Ключевые механизмы:**

1. **Адаптивная калибровка COVID19**
   - Если CNN вероятность < 0.35 → применить shift +0.10
   - Компенсирует систематический underestimation на некоторых датасетах

2. **Защита от артефактов**
   - Если KSL > 0.55 && CNN < 0.25 → clip KSL до 0.45
   - Высокий KSL при низком CNN может означать артефакты, а не патологию

3. **Взвешенная агрегация**
   - KSL weight: 60% (медицинская интерпретируемость)
   - CNN weight: 40% (паттерн-распознавание)
   - Hybrid prob = 0.6×KSL + 0.4×CNN

4. **Многокритериальное решение**
   - Комбинация: KSL positive + CNN > 0.40 → патология
   - High KSL alone: score > 0.47 → патология
   - Hybrid threshold: combined > 0.32 → патология

**Оптимизация:**
- Приоритет: **высокая чувствительность** (не пропустить патологию)
- Лучше false positive, чем пропустить заболевание (медицинский скрининг)

### Отключенные компоненты (для тестирования)

- ❌ **LUNA16 Nodule Detector** - 3D детекция узлов
- ❌ **Cancer Classifier** - классификация малигнизации узлов

*Причина отключения:* Высокий FPR на доброкачественных случаях (Spec=22%). Используются только в тестовых скриптах.

### Производительность

- ⏱️ **Время обработки:** ~50-60 секунд на случай (GPU)
- 🎯 **Точность на LCT dataset:** 100% (3/3 тестовых случаев)
- 💾 **Использование GPU:** ~1GB памяти (NVIDIA A30)
- 📊 **Val AUC:** 0.9711 (COVID19 classifier)

## 📦 Предварительные требования

### 1. Клонирование репозитория с весами моделей

Проект использует **Git LFS** для хранения модели COVID19 (~294МБ):

```bash
# Установка Git LFS (если еще не установлен)
# Ubuntu/Debian:
sudo apt install git-lfs

# Инициализация Git LFS
git lfs install

# Клонирование репозитория с LFS файлами
git clone https://github.com/nrm/chest-ct-ai-service.git
cd chest-ct-ai-service/api

# Проверка загрузки весов модели
ls -lh models/
# Должно показать:
# covid19_classifier_fold1_best_auc.pth (~294МБ)
```

### 2. Системные требования

**Протестированная конфигурация:**
- **ОС**: Ubuntu 22.04.5 LTS
- **GPU**: NVIDIA A30 (24GB)
- **Драйвер**: NVIDIA-SMI 570.172.08, Driver Version: 570.172.08
- **CUDA**: Version 12.8
- **Docker**: 20.10+ с nvidia-container-toolkit

**Минимальные требования:**
- Docker 20.10+
- 8GB RAM (режим GPU) или 16GB RAM (режим CPU)
- 10GB свободного места на диске

## 🚀 Быстрый запуск

### Вариант 1: Docker (Рекомендуется)

```bash
# Режим GPU (автоопределение)
./run-gpu.sh

# Режим CPU
./run-cpu.sh

# Ручной Docker Compose
docker-compose up --build

# API будет доступен по адресу http://localhost:8000
```

### Вариант 2: Локальная разработка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
python main.py
```

### Вариант 3: Пакетная обработка (без API)

Для обработки папки с ZIP файлами без запуска API сервера.

#### 3.1. Python скрипт (локально)

```bash
# Обработка всех ZIP файлов в папке
python batch_process.py --input /path/to/studies/

# Указать выходной файл
python batch_process.py --input /path/to/studies/ --output results.xlsx

# CSV формат
python batch_process.py --input /path/to/studies/ --output results.csv

# Параллельная обработка (2 воркера)
python batch_process.py --input /path/to/studies/ --workers 2
```

#### 3.2. Docker (изолированная среда)

```bash
# Простой запуск (использует ./input и ./output)
./run-batch.sh

# Указать папки
./run-batch.sh /path/to/studies /path/to/results

# Или напрямую через docker-compose
INPUT_DIR=/path/to/studies OUTPUT_DIR=/path/to/results \
  docker-compose -f docker-compose.batch.yml up
```

**Преимущества batch_process.py:**
- ✅ Не требует запуска API сервера
- ✅ Простой запуск для экспертов
- ✅ Автоматическая генерация имени выходного файла
- ✅ Детальный прогресс обработки
- ✅ Статистика по результатам
- ✅ Docker изоляция (воспроизводимость)

## 📡 Использование API

> **📖 Полная документация API:** См. **[API.md](API.md)** для детального описания всех эндпоинтов, параметров и примеров.

### Quick Start

**1. Загрузка файла:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@norma_anon.zip"
```

**2. Проверка статуса:**
```bash
curl "http://localhost:8000/tasks/{task_id}/status"
```

**3. Скачивание результата:**
```bash
curl -J -O "http://localhost:8000/tasks/{task_id}/result/excel"
```

### Основные эндпоинты

- `POST /upload` - Загрузка DICOM ZIP
- `GET /tasks/{task_id}/status` - Статус обработки
- `GET /tasks/{task_id}/result/excel` - Excel результат
- `GET /tasks/{task_id}/result/csv` - CSV результат
- `GET /health` - Health check

**Детали:** См. [API.md](API.md)

## 📊 Формат вывода

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

## ⚙️ Конфигурация

### Переменные окружения (API режим)

```bash
# Максимальное количество параллельных задач обработки
export MAX_CONCURRENT_JOBS=2

# Таймаут обработки в секундах
export PROCESSING_TIMEOUT=600

# Запуск API
python main.py
```

Или в docker-compose.yml:
```yaml
environment:
  - MAX_CONCURRENT_JOBS=2
  - PROCESSING_TIMEOUT=600
```

### Параметры batch_process.py

```bash
# Справка по параметрам
python batch_process.py --help

# Основные параметры:
#   --input, -i   : Папка с ZIP файлами (обязательно)
#   --output, -o  : Выходной файл .xlsx или .csv (опционально)
#   --workers, -w : Количество параллельных воркеров (по умолчанию: 1)
```

## 🔧 Алгоритм работы

1. **Загрузка DICOM** - Извлечение и валидация DICOM файлов из ZIP
2. **COVID19 Classifier** - Анализ 2D срезов с вниманием (ResNet50 MIL)
3. **KSL Z-profile** - Извлечение медицинских признаков из Z-профиля
4. **Hybrid Decision** - Комбинация CNN + KSL с адаптивной калибровкой:
   - Калибровка COVID19: +0.10 при prob < 0.35
   - Защита от агрессивного KSL: clip при z_score > 0.55 и COVID19 < 0.25
   - Гибридный порог: 0.32 для баланса Sensitivity/Specificity
5. **Генерация отчета** - Excel/CSV с результатами

## 📈 Производительность и метрики

### Производительность

- **Время обработки**: 50-60 секунд на случай (требование: ≤600с)
- **Использование памяти**: ~2-4ГБ RAM, ~1ГБ GPU
- **Пропускная способность**: ~60-70 исследований/час (1 GPU)

### Диагностические метрики (с 95% доверительными интервалами)

Результаты тестирования на валидационных наборах данных:

**LCT Dataset (n=3):**
- Accuracy: 100% (95% CI: 43.8% - 100%)
- Sensitivity: 100% (95% CI: 34.2% - 100%)
- Specificity: 100% (95% CI: 5.5% - 100%)

**COVID19 Dataset (n=60):**
- Accuracy: 50.0% (95% CI: 37.6% - 62.4%)
- Sensitivity: 60.0% (95% CI: 42.3% - 75.4%)
- Specificity: 40.0% (95% CI: 24.6% - 57.7%)

**Cancer Dataset (n=20):**
- Accuracy: 55.0% (95% CI: 33.2% - 75.0%)
- Sensitivity: 88.9% (95% CI: 56.5% - 98.0%)
- Specificity: 27.3% (95% CI: 9.0% - 56.6%)

> **Примечание:** Широкие доверительные интервалы обусловлены малым размером валидационных выборок. Система оптимизирована для медицинского скрининга с приоритетом на Sensitivity (минимизация пропущенных патологий).

**Ограничения:**
- Поддержка DICOM ZIP файлов до 1ГБ
- Максимум 2 одновременные задачи (настраивается)
- Таймаут: 10 минут на задачу (настраивается)

## 📁 Хранение данных (API режим)

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

## 🔍 Устранение неполадок

### Проблемы с Git LFS

Если веса моделей отсутствуют или повреждены:

```bash
# Проверка статуса Git LFS
git lfs status

# Повторная загрузка LFS файлов
git lfs pull

# Проверка размера файла модели
ls -lh models/
# covid19_classifier_fold1_best_auc.pth должен быть ~294МБ

# Если файл маленький (< 1КБ), это LFS указатель:
git lfs checkout
```

### Проблемы сборки Docker

**Ошибка**: `ModuleNotFoundError` или `FileNotFoundError` для моделей
```bash
# Убедитесь, что модель присутствует перед сборкой
ls -lh models/covid19_classifier_fold1_best_auc.pth
# Если отсутствует:
git lfs pull
```

### Низкая производительность

**Если обработка слишком медленная:**
- Проверьте использование GPU: должно быть ~1GB памяти
- Используйте `nvidia-smi` для мониторинга GPU
- При использовании CPU: обработка будет в 3-5 раз медленнее

### Ошибки обработки

**Проверка логов:**
```bash
# API режим
docker logs radiassist-api

# batch_process.py
# Логи выводятся в консоль
```

## 📄 Дополнительная документация

- **[API.md](API.md)** - Полная документация REST API
- **[BATCH_PROCESSING.md](BATCH_PROCESSING.md)** - Руководство по пакетной обработке
- **[BUILD.md](BUILD.md)** - Инструкции по сборке Docker образов
- **[main.py](main.py)** - API сервер (FastAPI)
- **[batch_process.py](batch_process.py)** - Скрипт пакетной обработки
- **[utils/metrics.py](utils/metrics.py)** - Метрики с 95% доверительными интервалами

## 🏥 Медицинские соображения

Система оптимизирована для медицинского скрининга:
- **Приоритет Sensitivity**: Лучше ложная тревога, чем пропущенная патология
- **Adaptive Calibration**: Корректирует систематические смещения модели
- **KSL Protection**: Защита от артефактов и ложных срабатываний
- **Hybrid Logic**: Комбинация CNN и медицинских правил

⚠️ **Внимание**: Система предназначена для помощи в принятии решений, а не замены врача-рентгенолога.

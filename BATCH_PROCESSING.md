# Batch Processing Guide

Руководство по пакетной обработке CT исследований с RadiAssist.

## 🎯 Варианты запуска

### 1. Python скрипт (локально)

Прямой запуск Python скрипта без Docker.

**Требования:**
- Python 3.8+
- Установленные зависимости (`pip install -r requirements.txt`)
- Веса модели в `models/`

**Использование:**
```bash
# Базовый запуск
python batch_process.py --input /path/to/studies/

# С указанием выходного файла
python batch_process.py --input /path/to/studies/ --output results.xlsx

# CSV формат
python batch_process.py --input /path/to/studies/ --output results.csv

# Параллельная обработка
python batch_process.py --input /path/to/studies/ --workers 2
```

**Преимущества:**
- ✅ Быстрый старт
- ✅ Легкая отладка
- ✅ Прямой доступ к файлам

**Недостатки:**
- ❌ Требует настройки окружения
- ❌ Зависимости от системы

---

### 2. Docker Compose (изолированная среда)

Запуск через Docker с автоматическим управлением volumes.

**Требования:**
- Docker 20.10+
- docker-compose 1.29+
- (Опционально) NVIDIA Docker для GPU

**Использование:**

#### 2.1. Через bash скрипт (рекомендуется)

```bash
# Простой запуск (использует ./input и ./output)
./run-batch.sh

# Указать свои папки
./run-batch.sh /path/to/studies /path/to/results

# Справка
./run-batch.sh --help
```

Скрипт автоматически:
- Проверяет наличие ZIP файлов
- Создает output директорию
- Определяет наличие GPU
- Запускает Docker с правильными volumes

#### 2.2. Напрямую через docker-compose

```bash
# Использовать переменные окружения
INPUT_DIR=/path/to/studies OUTPUT_DIR=/path/to/results \
  docker-compose -f docker-compose.batch.yml up

# Или экспортировать переменные
export INPUT_DIR=/path/to/studies
export OUTPUT_DIR=/path/to/results
docker-compose -f docker-compose.batch.yml up
```

**Преимущества:**
- ✅ Полная изоляция
- ✅ Воспроизводимость
- ✅ Не требует установки зависимостей
- ✅ GPU поддержка из коробки

**Недостатки:**
- ❌ Требует Docker
- ❌ Дольше первый запуск (сборка образа)

---

## 📊 Примеры использования

### Пример 1: Обработка тестовых данных

```bash
# Структура папок
studies/
├── norma_anon.zip
├── pneumonia_anon.zip
└── pneumotorax_anon.zip

# Запуск
./run-batch.sh ./studies ./results

# Результат
results/
└── radiassist_results_20251002_183045.xlsx
```

### Пример 2: Обработка большого датасета

```bash
# 100+ файлов
ls /data/chest_ct/*.zip | wc -l
# 123

# Запуск с 2 воркерами
python batch_process.py \
  --input /data/chest_ct \
  --output /data/results/batch_001.xlsx \
  --workers 2

# Время обработки: ~102 минуты (50s/case × 123 files / 2 workers)
```

### Пример 3: CSV выход для анализа

```bash
# Получить CSV для дальнейшей обработки в Python/R
python batch_process.py \
  --input ./studies \
  --output ./analysis/results.csv

# Анализ в Python
import pandas as pd
df = pd.read_csv('analysis/results.csv')
print(df.groupby('pathology').size())
```

---

## 📈 Ожидаемый вывод

### Консольный вывод

```
🔧 Initializing RadiAssist system...
Loading trained models...
✅ COVID19 classifier loaded from covid19_classifier_fold1_best_auc.pth
   Architecture: ResNet50 2D MIL (AUC: 0.9839)
⚠️  LUNA16 detector DISABLED (fast mode)
⚠️  Cancer classifier DISABLED (fast mode)
✅ Fixed KSL analyzer imported

✅ Models loaded:
   COVID19: YES
   LUNA16:  NO
   Cancer:  NO
   KSL:     YES

============================================================
[1/3] Processing: norma_anon.zip
============================================================

🔍 Testing case: norma_anon
  ✅ Ground truth: 0 (NORMAL)
  🔍 Validating input data...
     ✅ Valid chest CT (confidence: 1.000)
  🧠 Running COVID19 triage...
    ✅ Final pathology probability: 0.2431
  🧬 Running KSL Z-profile analysis...
    📊 Z-profile score: 0.2237
  🏥 Performing enhanced medical aggregation...
    🎯 Decision: 0 (NORMAL)

  ==================================================
  📋 FINAL RESULT:
     Prediction: 0 (NORMAL)
     Probability: 0.2217
     Method: hybrid_cnn_ksl
     Ground truth: 0 (NORMAL)
     ✅ CORRECT
  ==================================================

✅ Completed in 55.2s
   Pathology: 0 (prob: 0.222)

[2/3] Processing: pneumonia_anon.zip
...

============================================================
📊 BATCH PROCESSING SUMMARY
============================================================
Total files:     3
Successful:      3
Failed:          0
Total time:      172.5s (2.9 min)
Average/file:    57.5s

📈 Results:
   Normal:       1 (33.3%)
   Pathology:    2 (66.7%)

💾 Saving results to ./output/radiassist_results_20251002_183045.xlsx...
✅ Results saved successfully!
```

### Формат выходного файла

Excel/CSV содержит:

| path_to_study | study_uid | series_uid | probability_of_pathology | pathology | processing_status | time_of_processing | most_dangerous_pathology_type | pathology_localization |
|---------------|-----------|------------|-------------------------|-----------|-------------------|-------------------|------------------------------|----------------------|
| norma_anon.zip | 1.2.840... | 1.2.840... | 0.222 | 0 | SUCCESS | 55.2 | chest_abnormality | N/A |
| pneumonia_anon.zip | 1.2.840... | 1.2.840... | 0.993 | 1 | SUCCESS | 58.1 | chest_abnormality | 160,480,64,480,0,64 |
| pneumotorax_anon.zip | 1.2.840... | 1.2.840... | 0.270 | 0 | SUCCESS | 59.2 | chest_abnormality | N/A |

---

## ⚙️ Конфигурация

### Параметры командной строки

```bash
python batch_process.py --help
```

**Параметры:**
- `--input, -i` : Папка с ZIP файлами (обязательно)
- `--output, -o` : Выходной файл .xlsx/.csv (опционально, автогенерация)
- `--workers, -w` : Количество параллельных воркеров (по умолчанию: 1)

### Docker volumes

В `docker-compose.batch.yml`:

```yaml
volumes:
  # Read-only для входных данных (защита от изменений)
  - ${INPUT_DIR:-./input}:/data/input:ro
  
  # Read-write для выходных данных
  - ${OUTPUT_DIR:-./output}:/data/output
```

### Переменные окружения

```bash
# Папка с ZIP файлами
export INPUT_DIR=/path/to/studies

# Папка для результатов
export OUTPUT_DIR=/path/to/results

# Запуск
docker-compose -f docker-compose.batch.yml up
```

---

## 🐛 Устранение неполадок

### "No ZIP files found"

```bash
❌ No ZIP files found in /path/to/studies
```

**Решение:**
- Проверьте путь к папке
- Убедитесь что файлы имеют расширение `.zip`
- Проверьте permissions на папку

```bash
ls -la /path/to/studies/*.zip
```

### "Permission denied"

```bash
❌ Permission denied: /data/output
```

**Решение в Docker:**
```bash
# Создайте output папку с правильными permissions
mkdir -p output
chmod 777 output  # Или более строгие permissions

# Запустите снова
./run-batch.sh
```

### Медленная обработка

**CPU режим:**
- Ожидаемое время: ~180s на файл (в 3х медленнее GPU)

**GPU режим но медленно:**
```bash
# Проверьте использование GPU
nvidia-smi

# Должно показывать ~1GB GPU memory для radiassist-batch
```

### Out of Memory

```bash
❌ CUDA out of memory
```

**Решение:**
```bash
# Снизите количество воркеров до 1
python batch_process.py --input ./studies --workers 1

# Или увеличьте лимит памяти в docker-compose.batch.yml
```

---

## 📊 Производительность

### Время обработки

| Режим | Время/файл | Файлов/час |
|-------|------------|------------|
| GPU (1 worker) | ~55s | ~65 |
| GPU (2 workers) | ~30s | ~120 |
| CPU (1 worker) | ~180s | ~20 |

### Использование ресурсов

**GPU режим:**
- GPU Memory: ~1GB
- RAM: ~2-4GB
- CPU: 2-4 cores

**CPU режим:**
- RAM: ~4-8GB
- CPU: 100% (all cores)

---

## 🎯 Best Practices

1. **Проверьте входные данные:**
   ```bash
   ls -lh /path/to/studies/*.zip
   # Убедитесь что файлы не повреждены
   ```

2. **Используйте осмысленные имена выходных файлов:**
   ```bash
   python batch_process.py \
     --input ./hospital_a_batch_1 \
     --output ./results/hospital_a_2025_10_02.xlsx
   ```

3. **Сохраняйте логи:**
   ```bash
   python batch_process.py \
     --input ./studies \
     --output ./results/batch.xlsx 2>&1 | tee batch.log
   ```

4. **Параллелизация на больших датасетах:**
   ```bash
   # 2 воркера для GPU (оптимально)
   python batch_process.py --input ./big_dataset --workers 2
   ```

5. **Мониторинг в Docker:**
   ```bash
   # В отдельном терминале
   docker logs -f radiassist-batch
   ```

---

## 🔗 См. также

- [README.md](README.md) - Основная документация и системные требования
- [API.md](API.md) - REST API для интеграций
- [BUILD.md](BUILD.md) - Инструкции по сборке Docker образов

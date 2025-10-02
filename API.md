# RadiAssist REST API Documentation

Полное описание REST API для анализа КТ грудной клетки.

## 📡 Base URL

```
http://localhost:8000
```

## 🔐 Аутентификация

В текущей версии API не требует аутентификации. Для production развертывания рекомендуется добавить API ключи или OAuth2.

## 📋 Эндпоинты

### 1. Health Check

Проверка работоспособности сервиса.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "covid19": true,
    "luna16": false,
    "cancer": false,
    "ksl": true
  }
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. Upload DICOM Study

Загрузка ZIP-архива с DICOM файлами для анализа.

**Endpoint:** `POST /upload`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): ZIP archive with DICOM files

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "File uploaded successfully. Found 451 DICOM files.",
  "created_at": "2025-10-02T18:30:00.000Z",
  "filename": "norma_anon.zip"
}
```

**Status Codes:**
- `200 OK`: File uploaded successfully
- `400 Bad Request`: Invalid file format or missing file
- `413 Payload Too Large`: File size exceeds limit
- `429 Too Many Requests`: Processing queue is full

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@norma_anon.zip"
```

---

### 3. Get Task Status

Получение статуса обработки задачи.

**Endpoint:** `GET /tasks/{task_id}/status`

**Path Parameters:**
- `task_id` (required): UUID задачи

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2025-10-02T18:30:00.000Z",
  "completed_at": "2025-10-02T18:31:15.000Z",
  "processing_time": 75.3,
  "result_files": {
    "excel": "/tmp/radiassist_result.xlsx",
    "csv": "/tmp/radiassist_result.csv"
  }
}
```

**Status Values:**
- `pending`: Task created, waiting to start
- `processing`: Currently analyzing DICOM data
- `completed`: Analysis completed successfully
- `failed`: Error occurred during processing

**Status Codes:**
- `200 OK`: Task status retrieved
- `404 Not Found`: Task ID not found

**Example:**
```bash
curl "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/status"
```

---

### 4. Download Excel Results

Скачивание результатов в формате Excel (.xlsx).

**Endpoint:** `GET /tasks/{task_id}/result/excel`

**Path Parameters:**
- `task_id` (required): UUID задачи

**Response:** Binary file (application/vnd.openxmlformats-officedocument.spreadsheetml.sheet)

**Headers:**
- `Content-Disposition`: attachment; filename="norma_anon_result.xlsx"

**Status Codes:**
- `200 OK`: File download started
- `404 Not Found`: Task or file not found
- `425 Too Early`: Task not completed yet

**Example:**
```bash
# Download with original filename
curl -J -O "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/result/excel"

# Download with custom filename
curl -o "my_results.xlsx" "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/result/excel"
```

---

### 5. Download CSV Results

Скачивание результатов в формате CSV.

**Endpoint:** `GET /tasks/{task_id}/result/csv`

**Path Parameters:**
- `task_id` (required): UUID задачи

**Response:** Binary file (text/csv)

**Headers:**
- `Content-Disposition`: attachment; filename="norma_anon_result.csv"

**Status Codes:**
- `200 OK`: File download started
- `404 Not Found`: Task or file not found
- `425 Too Early`: Task not completed yet

**Example:**
```bash
curl -J -O "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/result/csv"
```

---

### 6. List Task Files

Получение списка всех файлов задачи.

**Endpoint:** `GET /tasks/{task_id}/files`

**Path Parameters:**
- `task_id` (required): UUID задачи

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "files": {
    "upload": {
      "path": "/data/uploads/550e8400-e29b-41d4-a716-446655440000/norma_anon.zip",
      "size": 45678912,
      "size_human": "43.5 MB"
    },
    "results": {
      "excel": {
        "path": "/data/results/550e8400-e29b-41d4-a716-446655440000/hackathon_test_results.xlsx",
        "size": 8192,
        "size_human": "8.0 KB"
      },
      "csv": {
        "path": "/data/results/550e8400-e29b-41d4-a716-446655440000/hackathon_test_results.csv",
        "size": 4096,
        "size_human": "4.0 KB"
      }
    }
  }
}
```

**Status Codes:**
- `200 OK`: Files list retrieved
- `404 Not Found`: Task not found

**Example:**
```bash
curl "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/files"
```

---

### 7. List All Tasks

Получение списка всех задач.

**Endpoint:** `GET /tasks`

**Query Parameters:**
- `limit` (optional): Maximum number of tasks to return (default: 100)
- `status` (optional): Filter by status (pending|processing|completed|failed)

**Response:**
```json
{
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2025-10-02T18:30:00.000Z",
      "filename": "norma_anon.zip"
    },
    {
      "task_id": "660f9511-f39c-52e5-b827-557766551111",
      "status": "processing",
      "created_at": "2025-10-02T18:35:00.000Z",
      "filename": "pneumonia_anon.zip"
    }
  ],
  "total": 2
}
```

**Status Codes:**
- `200 OK`: Tasks list retrieved

**Example:**
```bash
# Get all tasks
curl "http://localhost:8000/tasks"

# Get only completed tasks
curl "http://localhost:8000/tasks?status=completed"

# Limit to 10 tasks
curl "http://localhost:8000/tasks?limit=10"
```

---

### 8. Browse Data Directory

Просмотр структуры директории данных (для debugging).

**Endpoint:** `GET /data/browse`

**Query Parameters:**
- `path` (optional): Subdirectory to browse (default: root)

**Response:**
```json
{
  "path": "/data",
  "directories": ["uploads", "results"],
  "files": []
}
```

**Status Codes:**
- `200 OK`: Directory listing retrieved
- `404 Not Found`: Directory not found

**Example:**
```bash
# Browse root
curl "http://localhost:8000/data/browse"

# Browse uploads
curl "http://localhost:8000/data/browse?path=uploads"
```

---

## 📊 Output Format

Результаты анализа содержат следующие поля:

### Excel/CSV Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `path_to_study` | String | Original ZIP filename | "norma_anon.zip" |
| `study_uid` | String | DICOM Study Instance UID | "1.2.840.113..." |
| `series_uid` | String | DICOM Series Instance UID | "1.2.840.113..." |
| `probability_of_pathology` | Float | Pathology probability (0.0-1.0) | 0.243 |
| `pathology` | Integer | Binary classification (0=normal, 1=pathology) | 0 |
| `processing_status` | String | Processing status | "SUCCESS" |
| `time_of_processing` | Float | Processing time in seconds | 55.3 |
| `most_dangerous_pathology_type` | String | Type of detected pathology | "chest_abnormality" |
| `pathology_localization` | String | Bounding box coordinates | "160,480,64,480,0,160" |

### Pathology Localization Format

Coordinates in format: `x_min,x_max,y_min,y_max,z_min,z_max`

- `x_min, x_max`: Horizontal bounds (pixels)
- `y_min, y_max`: Vertical bounds (pixels)
- `z_min, z_max`: Depth bounds (slices)

---

## ⚠️ Error Handling

### Error Response Format

```json
{
  "error": "Error description",
  "detail": "Detailed error message",
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Common Errors

**400 Bad Request**
```json
{
  "error": "Invalid file format",
  "detail": "Only ZIP archives are supported"
}
```

**404 Not Found**
```json
{
  "error": "Task not found",
  "detail": "Task ID 550e8400-... does not exist"
}
```

**413 Payload Too Large**
```json
{
  "error": "File too large",
  "detail": "Maximum file size is 1GB"
}
```

**429 Too Many Requests**
```json
{
  "error": "Queue full",
  "detail": "Maximum 2 concurrent tasks allowed. Please try again later."
}
```

**500 Internal Server Error**
```json
{
  "error": "Processing failed",
  "detail": "DICOM parsing error: Invalid file format"
}
```

---

## 🔄 Workflow Example

### Complete Processing Workflow

```bash
# 1. Upload file
TASK_ID=$(curl -X POST "http://localhost:8000/upload" \
  -F "file=@norma_anon.zip" | jq -r '.task_id')

echo "Task ID: $TASK_ID"

# 2. Poll status until completed
while true; do
  STATUS=$(curl "http://localhost:8000/tasks/$TASK_ID/status" | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi

  sleep 5
done

# 3. Download results if completed
if [ "$STATUS" = "completed" ]; then
  curl -J -O "http://localhost:8000/tasks/$TASK_ID/result/excel"
  echo "Results downloaded!"
else
  echo "Processing failed!"
  exit 1
fi
```

---

## ⚙️ Configuration

### Environment Variables

Configure API behavior via environment variables:

```bash
# Maximum concurrent processing tasks
export MAX_CONCURRENT_JOBS=2

# Processing timeout in seconds
export PROCESSING_TIMEOUT=600

# Data directory
export DATA_DIR=/data

# API host and port
export HOST=0.0.0.0
export PORT=8000
```

### Performance Tuning

**For high throughput:**
```bash
MAX_CONCURRENT_JOBS=4  # Increase parallel processing
PROCESSING_TIMEOUT=900  # Allow longer processing time
```

**For limited resources:**
```bash
MAX_CONCURRENT_JOBS=1  # Sequential processing only
PROCESSING_TIMEOUT=600  # Standard timeout
```

---

## 📈 Rate Limiting

Current implementation:
- Maximum concurrent tasks: 2 (configurable)
- Queue size: Unlimited (tasks wait in queue)
- Processing timeout: 10 minutes per task

When queue is full:
- HTTP 429 returned
- Retry-After header suggests wait time
- Clients should implement exponential backoff

---

## 🔒 Security Considerations

**Current Implementation (Development):**
- No authentication required
- All endpoints publicly accessible
- File size limited to 1GB

**Production Recommendations:**
- Add API key authentication
- Implement rate limiting per user
- Add HTTPS/TLS encryption
- Sanitize file uploads
- Add input validation
- Implement audit logging

---

## 📞 Support

For API issues or questions:
- Check logs: `docker logs radiassist-api`
- Review documentation: [README.md](README.md)
- Batch processing: [BATCH_PROCESSING.md](BATCH_PROCESSING.md)

---

## 📝 Changelog

### Version 1.0.0 (2025-10-02)
- Initial API release
- Support for DICOM ZIP uploads
- Excel/CSV output format
- Async processing with task queue
- Docker containerization

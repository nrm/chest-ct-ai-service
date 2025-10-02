#!/usr/bin/env python3
"""
RadiAssist Chest CT Analysis API
Simple REST API for hackathon compliance
"""

import os
import sys
import uuid
import asyncio
import zipfile
import threading
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
import uvicorn

# Add local modules to path
sys.path.append(str(Path(__file__).parent))

from hackathon.tester import HackathonTester
from hackathon.reporting import create_excel_output

# GPU Diagnostics at startup
def check_gpu_availability():
    print("🔍 GPU Diagnostics at startup:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi command successful:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi failed:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ nvidia-smi command timed out")
    except FileNotFoundError:
        print("❌ nvidia-smi command not found")
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")

    # Check PyTorch GPU availability
    try:
        import torch
        print(f"🔥 PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔥 PyTorch CUDA device count: {torch.cuda.device_count()}")
            print(f"🔥 PyTorch current device: {torch.cuda.current_device()}")
            print(f"🔥 PyTorch device name: {torch.cuda.get_device_name()}")
        else:
            print("❌ PyTorch CUDA not available")
    except Exception as e:
        print(f"❌ PyTorch CUDA check error: {e}")

check_gpu_availability()

# Configuration
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "600"))  # 10 minutes

app = FastAPI(
    title="RadiAssist Chest CT API",
    description="AI service for chest CT pathology classification",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production лучше указать конкретные origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories
DATA_DIR = Path("./data")
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

# Create directories on startup
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Task storage (in production, use Redis/database)
tasks: Dict[str, Dict] = {}

# Thread pool for heavy processing
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)

# Semaphore to limit concurrent processing
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_files: Optional[Dict[str, str]] = None
    segmentation: Optional[Dict] = None  # Информация о сегментации
    result_data: Optional[Dict] = None   # Результаты анализа
    immediate_slices: Optional[Dict] = None  # Немедленные слайсы

class UploadResponse(BaseModel):
    task_id: str
    status: str
    message: str

class UploadRequest(BaseModel):
    processing_timeout: Optional[int] = 600  # Default 10 minutes

def process_dicom_zip_sync(task_id: str, zip_path: Path, output_dir: Path, timeout: int = PROCESSING_TIMEOUT):
    """Synchronous processing function - runs in separate thread"""
    try:
        print(f"🔄 Processing task {task_id}: {zip_path}")
        print(f"📁 Output directory: {output_dir}")

        # 1. НЕМЕДЛЕННАЯ ГЕНЕРАЦИЯ PNG СЛАЙСОВ для просмотра (САМОЕ ПЕРВОЕ!)
        immediate_slices_info = None
        print(f"📸 Начинаем генерацию немедленных слайсов для задачи {task_id}...")
        try:
            from utils.immediate_slices import generate_immediate_slices, extract_dicom_files_from_zip
            
            print(f"📸 Генерация немедленных PNG слайсов для задачи {task_id}...")
            dicom_files = extract_dicom_files_from_zip(zip_path)
            print(f"📁 Извлечено DICOM файлов: {len(dicom_files) if dicom_files else 0}")
            
            if dicom_files and len(dicom_files) > 0:
                immediate_slices_info = generate_immediate_slices(task_id, dicom_files, output_dir, max_slices=1000, slice_step=3)  # Каждый 3-й слайс
                if immediate_slices_info and 'generated_slices' in immediate_slices_info:
                    print(f"✅ Немедленные слайсы сгенерированы: {immediate_slices_info['generated_slices']} слайсов")
                    
                    # Обновляем задачу с немедленными слайсами
                    tasks[task_id]["immediate_slices"] = immediate_slices_info
                    print(f"📊 Задача обновлена с немедленными слайсами")
                else:
                    print(f"⚠️  Немедленные слайсы не сгенерированы корректно")
            else:
                print(f"⚠️  DICOM файлы не найдены в ZIP для немедленных слайсов")
                
        except Exception as immediate_error:
            print(f"⚠️  Немедленные слайсы не сгенерированы: {immediate_error}")

        # 2. СЕГМЕНТАЦИЯ
        segmentation_metadata = None
        try:
            from utils.segmentation_wrapper import run_segmentation
            
            # Extract DICOM directory from ZIP
            import zipfile
            import tempfile
            temp_dicom_dir = Path(tempfile.mkdtemp(prefix=f"dicom_{task_id}_"))
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dicom_dir)
                
                # Находим директорию с DICOM файлами
                dicom_dir = temp_dicom_dir
                for root, dirs, files in os.walk(temp_dicom_dir):
                    if any(f.lower().endswith('.dcm') or (not '.' in f and os.path.isfile(os.path.join(root, f))) for f in files):
                        dicom_dir = Path(root)
                        break
                
                # Запускаем сегментацию (только body и lungs по умолчанию)
                print(f"🧠 Starting segmentation for task {task_id}...")
                segmentation_metadata = run_segmentation(task_id, dicom_dir, output_dir, include_bones=False)
                
                if segmentation_metadata:
                    print(f"✅ Segmentation completed for task {task_id}")
                    print(f"📊 Segmentation metadata keys: {list(segmentation_metadata.keys())}")
                    
                    # Обновляем задачу с информацией о сегментации СРАЗУ
                    tasks[task_id]["segmentation"] = segmentation_metadata
                    print(f"✅ Task {task_id} updated with segmentation data")
                else:
                    print(f"⚠️ Segmentation metadata is empty for task {task_id}")
                
            finally:
                # Очищаем временную директорию
                import shutil
                shutil.rmtree(temp_dicom_dir, ignore_errors=True)
                
        except Exception as seg_error:
            print(f"⚠️  Segmentation failed for task {task_id}, continuing without masks: {seg_error}")

        # 3. ГЕНЕРАЦИЯ РАСКРАШЕННЫХ СЛАЙСОВ (если сегментация прошла успешно)
        if segmentation_metadata:
            try:
                print(f"🎨 Генерация раскрашенных слайсов для задачи {task_id}...")
                # Здесь будет логика генерации раскрашенных слайсов с масками
                # Пока что используем существующую логику из segmentation_wrapper
                print(f"✅ Раскрашенные слайсы сгенерированы")
            except Exception as colored_error:
                print(f"⚠️  Раскрашенные слайсы не сгенерированы: {colored_error}")

        # Initialize hackathon tester
        tester = HackathonTester(disable_validation=False)  # Enable validation

        # Process single case
        result = tester.test_single_case(str(zip_path))
        
        if result is None:
            raise ValueError("Test returned None result")

        # Create Excel/CSV output in permanent location
        excel_path, csv_path = create_excel_output([result], str(output_dir))

        # Check if validation failed
        is_validation_failed = result.get("status") == "validation_failed"
        
        # Update task with results FIRST (before segmentation)
        task_update = {
            "status": "completed",  # Always mark as completed, even if validation failed
            "completed_at": datetime.now().isoformat(),
            "result_files": {
                "excel": str(excel_path),
                "csv": str(csv_path)
            },
            "result_data": {
                "probability_of_pathology": result.get("probability_of_pathology", 0.5),
                "pathology": result.get("pathology", 0),
                "processing_time": result.get("processing_time", 0),
                "status": result.get("status", "UNKNOWN")
            }
        }
        
        # Add validation error if applicable
        if is_validation_failed:
            task_update["validation_error"] = result.get("error", "Data validation failed")
            print(f"⚠️  Task {task_id} completed with validation error: {task_update['validation_error']}")
        else:
            print(f"✅ Task {task_id} completed successfully")
        
        # Add segmentation results to task update
        if segmentation_metadata:
            task_update["segmentation"] = segmentation_metadata
        
        # Add immediate slices info to task update
        if immediate_slices_info:
            task_update["immediate_slices"] = immediate_slices_info
        
        tasks[task_id].update(task_update)
        print(f"📄 Results saved: {excel_path.name}, {csv_path.name}")

    except Exception as e:
        print(f"❌ Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()
        tasks[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error_message": str(e)
        })

async def process_dicom_zip(task_id: str, zip_path: Path, output_dir: Path, timeout: int = PROCESSING_TIMEOUT):
    """Async wrapper that runs heavy processing in thread pool with concurrency limit"""
    async with processing_semaphore:
        # Update task status to processing when slot becomes available
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["started_at"] = datetime.now().isoformat()

        loop = asyncio.get_event_loop()

        try:
            # Run with timeout
            await asyncio.wait_for(
                loop.run_in_executor(executor, process_dicom_zip_sync, task_id, zip_path, output_dir, timeout),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            print(f"⏰ Task {task_id} timed out after {timeout} seconds")
            tasks[task_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error_message": f"Processing timed out after {timeout} seconds"
            })

@app.post("/upload", response_model=UploadResponse)
async def upload_dicom_zip(
    file: UploadFile = File(...),
    processing_timeout: int = 600
):
    """Upload DICOM ZIP file for processing"""
    print(f"📤 Upload started: {file.filename}")
    
    # Validate file
    if not file.filename.endswith('.zip'):
        print(f"❌ Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only ZIP files are accepted")

    # Check if processing queue is full
    processing_count = sum(1 for task in tasks.values() if task["status"] in ["pending", "processing"])
    if processing_count >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429,
            detail=f"Processing queue is full. Maximum {MAX_CONCURRENT_JOBS} concurrent jobs allowed. Try again later."
        )

    # Generate task ID
    task_id = str(uuid.uuid4())
    print(f"📝 Created task: {task_id}")

    # Create task record
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "filename": file.filename
    }

    try:
        print(f"💾 Saving file for task {task_id}...")
        # Create permanent directories for this task
        task_upload_dir = UPLOADS_DIR / task_id
        task_results_dir = RESULTS_DIR / task_id
        task_upload_dir.mkdir(exist_ok=True)
        task_results_dir.mkdir(exist_ok=True)

        # Save uploaded file permanently
        zip_path = task_upload_dir / file.filename
        with open(zip_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Validate ZIP file
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                file_list = z.namelist()
                print(f"📂 Files in ZIP: {len(file_list)}")
                print(f"📂 First 10 files: {file_list[:10]}")
                
                # DICOM files can have .dcm extension or no extension at all
                # Also check for common patterns in medical imaging
                dicom_files = [
                    f for f in file_list 
                    if (
                        f.endswith('.dcm') or 
                        f.endswith('.DCM') or
                        f.endswith('.dicom') or
                        # Files without extension (common in DICOM datasets)
                        ('.' not in os.path.basename(f) and not f.endswith('/') and len(os.path.basename(f)) > 0)
                    )
                ]
                
                print(f"🔍 Found {len(dicom_files)} potential DICOM files")
                
                if len(dicom_files) == 0:
                    # More lenient: accept any non-directory files
                    dicom_files = [f for f in file_list if not f.endswith('/') and len(os.path.basename(f)) > 0]
                    print(f"⚠️  No obvious DICOM files, accepting all {len(dicom_files)} files in archive")
                
                if len(dicom_files) == 0:
                    raise HTTPException(status_code=400, detail="No DICOM files found in ZIP")
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")

        # Store paths in task
        tasks[task_id].update({
            "zip_path": str(zip_path),
            "upload_dir": str(task_upload_dir),
            "results_dir": str(task_results_dir),
            "dicom_count": len(dicom_files)
        })

        # Start background processing in thread pool (non-blocking)
        asyncio.create_task(process_dicom_zip(task_id, zip_path, task_results_dir, processing_timeout))

        return UploadResponse(
            task_id=task_id,
            status="pending",
            message=f"File uploaded successfully. Found {len(dicom_files)} DICOM files."
        )

    except HTTPException:
        # Re-raise HTTP exceptions (400, 429, etc.)
        if task_id in tasks:
            del tasks[task_id]
        raise
    except Exception as e:
        # Log unexpected errors
        print(f"❌ Upload error for task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        if task_id in tasks:
            del tasks[task_id]
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/tasks/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get task status by ID"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = tasks[task_id]

    return TaskStatus(
        task_id=task_id,
        status=task_data["status"],
        created_at=task_data["created_at"],
        completed_at=task_data.get("completed_at"),
        error_message=task_data.get("error_message"),
        result_files=task_data.get("result_files"),
        segmentation=task_data.get("segmentation"),
        result_data=task_data.get("result_data"),
        immediate_slices=task_data.get("immediate_slices")
    )

@app.get("/tasks/{task_id}/result/excel")
async def download_excel_result(task_id: str):
    """Download Excel result file"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    excel_path = task["result_files"]["excel"]
    if not Path(excel_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    # Use original filename without extension + _result
    original_filename = Path(task["filename"]).stem  # Remove .zip extension
    result_filename = f"{original_filename}_result.xlsx"

    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=result_filename
    )

@app.get("/tasks/{task_id}/result/csv")
async def download_csv_result(task_id: str):
    """Download CSV result file"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    csv_path = task["result_files"]["csv"]
    if not Path(csv_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    # Use original filename without extension + _result
    original_filename = Path(task["filename"]).stem  # Remove .zip extension
    result_filename = f"{original_filename}_result.csv"

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=result_filename
    )

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return {"tasks": list(tasks.values())}

@app.get("/tasks/{task_id}/files")
async def list_task_files(task_id: str):
    """List all files for a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    files_info = {
        "task_id": task_id,
        "upload_files": [],
        "result_files": []
    }

    # List uploaded files
    if "upload_dir" in task:
        upload_dir = Path(task["upload_dir"])
        if upload_dir.exists():
            files_info["upload_files"] = [f.name for f in upload_dir.iterdir() if f.is_file()]

    # List result files
    if "results_dir" in task:
        results_dir = Path(task["results_dir"])
        if results_dir.exists():
            files_info["result_files"] = [f.name for f in results_dir.iterdir() if f.is_file()]

    return files_info

@app.get("/data/browse")
async def browse_data_directory():
    """Browse data directory structure"""
    def scan_directory(path: Path) -> dict:
        if not path.exists():
            return {"error": "Directory not found"}

        items = []
        for item in path.iterdir():
            if item.is_dir():
                items.append({
                    "name": item.name,
                    "type": "directory",
                    "size": len(list(item.iterdir())) if item.is_dir() else 0
                })
            else:
                items.append({
                    "name": item.name,
                    "type": "file",
                    "size": item.stat().st_size
                })

        return {
            "path": str(path),
            "items": sorted(items, key=lambda x: (x["type"], x["name"]))
        }

    return {
        "data_directory": scan_directory(DATA_DIR),
        "uploads": scan_directory(UPLOADS_DIR),
        "results": scan_directory(RESULTS_DIR)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Count processing tasks
    processing_count = sum(1 for task in tasks.values() if task["status"] == "processing")

    return {
        "status": "healthy",
        "service": "radiassist-api",
        "version": "1.0.0",
        "config": {
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "processing_timeout": PROCESSING_TIMEOUT,
            "current_processing": processing_count,
            "available_slots": MAX_CONCURRENT_JOBS - processing_count
        }
    }

@app.get("/gpu-status")
async def gpu_status():
    """GPU status and diagnostics endpoint"""
    gpu_info = {
        "nvidia_smi_available": False,
        "nvidia_smi_output": None,
        "pytorch_cuda_available": False,
        "pytorch_cuda_device_count": 0,
        "pytorch_current_device": None,
        "pytorch_device_name": None,
        "cupy_available": False,
        "cupy_device_count": 0,
        "cupy_memory_total": 0,
        "cupy_memory_free": 0,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "not_set"),
        "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES", "not_set"),
        "nvidia_driver_capabilities": os.getenv("NVIDIA_DRIVER_CAPABILITIES", "not_set")
    }
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info["nvidia_smi_available"] = True
            gpu_info["nvidia_smi_output"] = result.stdout
        else:
            gpu_info["nvidia_smi_output"] = f"Error: {result.stderr}"
    except Exception as e:
        gpu_info["nvidia_smi_output"] = f"Exception: {str(e)}"
    
    # Check PyTorch CUDA
    try:
        import torch
        gpu_info["pytorch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info["pytorch_cuda_device_count"] = torch.cuda.device_count()
            gpu_info["pytorch_current_device"] = torch.cuda.current_device()
            gpu_info["pytorch_device_name"] = torch.cuda.get_device_name()
    except Exception as e:
        gpu_info["pytorch_error"] = str(e)
    
    # Check CuPy (for GPU segmentation)
    try:
        import cupy as cp
        gpu_info["cupy_available"] = True
        gpu_info["cupy_device_count"] = cp.cuda.runtime.getDeviceCount()
        if gpu_info["cupy_device_count"] > 0:
            memory_info = cp.cuda.runtime.memGetInfo()
            gpu_info["cupy_memory_free"] = memory_info[0] // (1024**3)  # Free memory in GB
            gpu_info["cupy_memory_total"] = memory_info[1] // (1024**3)  # Total memory in GB
    except Exception as e:
        gpu_info["cupy_error"] = str(e)
    
    return gpu_info

@app.get("/tasks/{task_id}/slices")
async def get_task_slices(task_id: str):
    """Get list of available DICOM slices for a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    # Find DICOM directory in upload folder
    upload_dir = DATA_DIR / "uploads" / task_id
    
    dicom_files = []
    for zip_file in upload_dir.glob("*.zip"):
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find DICOM directory
                from utils.dicom_to_image import get_dicom_files_sorted
                temp_path = Path(temp_dir)
                
                # Look for DICOM files recursively
                for root, dirs, files in os.walk(temp_path):
                    dicom_dir = Path(root)
                    sorted_files = get_dicom_files_sorted(dicom_dir)
                    if sorted_files:
                        dicom_files = [{"index": i, "filename": f.name} for i, f in enumerate(sorted_files)]
                        break
        except Exception as e:
            print(f"Error listing DICOM files: {e}")
    
    return {
        "task_id": task_id,
        "slice_count": len(dicom_files),
        "slices": dicom_files
    }


@app.get("/tasks/{task_id}/slices/{slice_index}")
async def get_task_slice_image(task_id: str, slice_index: int):
    """Get a specific DICOM slice as PNG image"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    # Find and convert DICOM slice
    upload_dir = DATA_DIR / "uploads" / task_id
    
    for zip_file in upload_dir.glob("*.zip"):
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                from utils.dicom_to_image import get_dicom_files_sorted, dicom_to_png_bytes
                temp_path = Path(temp_dir)
                
                # Find DICOM directory
                for root, dirs, files in os.walk(temp_path):
                    dicom_dir = Path(root)
                    sorted_files = get_dicom_files_sorted(dicom_dir)
                    
                    if sorted_files and 0 <= slice_index < len(sorted_files):
                        dicom_file = sorted_files[slice_index]
                        
                        # Get bounding box if available
                        bbox = None
                        if "result_data" in task:
                            loc_str = task["result_data"].get("pathology_localization")
                            if loc_str and loc_str != "":
                                try:
                                    bbox = tuple(map(float, loc_str.split(',')))
                                except:
                                    pass
                        
                        # Convert to PNG
                        from utils.dicom_to_image import get_slice_with_bounding_box
                        png_bytes = get_slice_with_bounding_box(dicom_file, bbox)
                        
                        return Response(content=png_bytes, media_type="image/png")
        except Exception as e:
            print(f"Error getting slice image: {e}")
            import traceback
            traceback.print_exc()
    
    raise HTTPException(status_code=404, detail="Slice not found")


@app.get("/tasks/{task_id}/segmentation/metadata")
async def get_segmentation_metadata(task_id: str):
    """Get segmentation metadata for a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    segmentation = task.get("segmentation")
    if not segmentation:
        raise HTTPException(status_code=404, detail="Segmentation not available for this task")
    
    return segmentation


@app.get("/tasks/{task_id}/segmentation/mask/{component}")
async def get_segmentation_mask_3d(task_id: str, component: str):
    """Get 3D mask for a specific component as numpy array"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    segmentation = task.get("segmentation")
    if not segmentation or "components" not in segmentation:
        raise HTTPException(status_code=404, detail="Segmentation not available")
    
    if component not in segmentation["components"]:
        raise HTTPException(status_code=404, detail=f"Component '{component}' not found")
    
    # Get mask file path
    comp_data = segmentation["components"][component]
    mask_file = comp_data.get("mask_3d_file")
    
    if not mask_file:
        raise HTTPException(status_code=404, detail="3D mask file not found")
    
    output_dir = DATA_DIR / "results" / task_id
    mask_path = output_dir / mask_file
    
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Mask file does not exist")
    
    # Load numpy array and convert to bytes
    import numpy as np
    mask = np.load(str(mask_path))
    
    # Convert to JSON-compatible format
    mask_data = {
        "shape": list(mask.shape),
        "data": mask.flatten().tolist(),
        "dtype": str(mask.dtype)
    }
    
    return JSONResponse(content=mask_data)


@app.get("/tasks/{task_id}/segmentation/preview")
async def get_segmentation_preview(task_id: str):
    """Get segmentation preview image"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    output_dir = DATA_DIR / "results" / task_id
    preview_path = output_dir / "segmentation_preview.png"
    
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview image not found")
    
    return FileResponse(
        preview_path,
        media_type="image/png",
        filename=f"segmentation_preview_{task_id}.png"
    )


@app.get("/tasks/{task_id}/segmentation/slices")
async def get_mask_slices_list(task_id: str):
    """Get list of available mask slices"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    segmentation = task.get("segmentation")
    if not segmentation:
        raise HTTPException(status_code=404, detail="Segmentation not available for this task")
    
    # Find slices directory
    output_dir = DATA_DIR / "results" / task_id
    slices_dir = output_dir / "masks" / "mask_slices"
    slices_info_path = slices_dir / "slices_info.json"
    
    if not slices_info_path.exists():
        raise HTTPException(status_code=404, detail="Mask slices not generated yet")
    
    try:
        with open(slices_info_path, 'r') as f:
            slices_info = json.load(f)
        return slices_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading slices info: {str(e)}")


@app.get("/tasks/{task_id}/segmentation/slices/{slice_filename}")
async def get_mask_slice_image(task_id: str, slice_filename: str):
    """Get specific mask slice image"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    segmentation = task.get("segmentation")
    if not segmentation:
        raise HTTPException(status_code=404, detail="Segmentation not available for this task")
    
    # Find slice image
    output_dir = DATA_DIR / "results" / task_id
    slices_dir = output_dir / "masks" / "mask_slices"
    slice_path = slices_dir / slice_filename
    
    if not slice_path.exists():
        raise HTTPException(status_code=404, detail="Slice image not found")
    
    return FileResponse(
        slice_path,
        media_type="image/png",
        filename=slice_filename
    )


@app.get("/tasks/{task_id}/immediate-slices")
async def get_immediate_slices_list(task_id: str):
    """Get list of immediate slices (generated before segmentation)"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    # Немедленные слайсы доступны сразу после создания задачи
    # if task["status"] not in ["pending", "processing", "completed"]:
    #     raise HTTPException(status_code=400, detail="Task not available yet")
    
    immediate_slices = task.get("immediate_slices")
    if not immediate_slices:
        raise HTTPException(status_code=404, detail="Immediate slices not generated yet")
    
    return immediate_slices


@app.get("/tasks/{task_id}/immediate-slices/{slice_filename}")
async def get_immediate_slice_image(task_id: str, slice_filename: str):
    """Get specific immediate slice image"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    # Немедленные слайсы доступны сразу после создания задачи
    # if task["status"] not in ["pending", "processing", "completed"]:
    #     raise HTTPException(status_code=400, detail="Task not available yet")
    
    immediate_slices = task.get("immediate_slices")
    if not immediate_slices:
        raise HTTPException(status_code=404, detail="Immediate slices not generated yet")
    
    # Find slice image
    output_dir = DATA_DIR / "results" / task_id
    slices_dir = output_dir / "immediate_slices"
    slice_path = slices_dir / slice_filename
    
    if not slice_path.exists():
        raise HTTPException(status_code=404, detail="Slice image not found")
    
    return FileResponse(
        slice_path,
        media_type="image/png",
        filename=slice_filename
    )


@app.post("/tasks/{task_id}/segmentation/bones")
async def run_bones_segmentation(task_id: str):
    """Run bones segmentation for a completed task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    # Check if bones segmentation already exists
    if "segmentation" in task and "components" in task["segmentation"]:
        if "bone" in task["segmentation"]["components"] or "spine" in task["segmentation"]["components"]:
            return {"message": "Bones segmentation already exists", "status": "already_exists"}
    
    try:
        from utils.segmentation_wrapper import run_segmentation
        
        # Get the original ZIP file path
        zip_path = Path(task["zip_file"])
        if not zip_path.exists():
            raise HTTPException(status_code=404, detail="Original ZIP file not found")
        
        # Extract DICOM directory from ZIP
        import zipfile
        import tempfile
        temp_dicom_dir = Path(tempfile.mkdtemp(prefix=f"dicom_{task_id}_bones_"))
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dicom_dir)
            
            # Находим директорию с DICOM файлами
            dicom_dir = temp_dicom_dir
            for root, dirs, files in os.walk(temp_dicom_dir):
                if any(f.lower().endswith('.dcm') or (not '.' in f and os.path.isfile(os.path.join(root, f))) for f in files):
                    dicom_dir = Path(root)
                    break
            
            # Get output directory
            output_dir = Path(task["output_dir"])
            
            # Запускаем сегментацию с костями (включая ct_lung.py)
            print(f"🦴 Starting enhanced bones segmentation for task {task_id}...")
            bones_segmentation = run_segmentation(task_id, dicom_dir, output_dir, include_bones=True)
            
            if bones_segmentation:
                # Merge with existing segmentation or create new
                if "segmentation" in task:
                    # Merge bones components into existing segmentation
                    for comp_name, comp_data in bones_segmentation["components"].items():
                        if comp_name in ["bone", "spine", "ribs"]:
                            task["segmentation"]["components"][comp_name] = comp_data
                else:
                    # Create new segmentation entry
                    task["segmentation"] = bones_segmentation
                
                print(f"✅ Bones segmentation completed for task {task_id}")
                return {"message": "Bones segmentation completed", "status": "success"}
            else:
                return {"message": "Bones segmentation failed", "status": "failed"}
                
        finally:
            # Очищаем временную директорию
            import shutil
            shutil.rmtree(temp_dicom_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Bones segmentation failed for task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Bones segmentation failed: {str(e)}")


@app.get("/")
async def root():
    """API information"""
    return {
        "service": "RadiAssist Chest CT Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload DICOM ZIP file",
            "status": "GET /tasks/{task_id}/status - Get task status",
            "excel": "GET /tasks/{task_id}/result/excel - Download Excel result",
            "csv": "GET /tasks/{task_id}/result/csv - Download CSV result",
            "tasks": "GET /tasks - List all tasks",
            "slices": "GET /tasks/{task_id}/slices - Get list of DICOM slices",
            "slice_image": "GET /tasks/{task_id}/slices/{index} - Get slice as PNG",
            "segmentation_meta": "GET /tasks/{task_id}/segmentation/metadata - Get segmentation metadata",
            "segmentation_mask": "GET /tasks/{task_id}/segmentation/mask/{component} - Get 3D mask",
            "segmentation_preview": "GET /tasks/{task_id}/segmentation/preview - Get segmentation preview",
            "segmentation_slices": "GET /tasks/{task_id}/segmentation/slices - Get list of mask slices",
            "segmentation_slice": "GET /tasks/{task_id}/segmentation/slices/{filename} - Get mask slice image",
            "immediate_slices": "GET /tasks/{task_id}/immediate-slices - Get list of immediate slices",
            "immediate_slice": "GET /tasks/{task_id}/immediate-slices/{filename} - Get immediate slice image",
            "bones_segmentation": "POST /tasks/{task_id}/segmentation/bones - Run bones segmentation",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
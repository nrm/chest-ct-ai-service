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
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
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

class UploadResponse(BaseModel):
    task_id: str
    status: str
    message: str

def process_dicom_zip_sync(task_id: str, zip_path: Path, output_dir: Path):
    """Synchronous processing function - runs in separate thread"""
    try:
        print(f"🔄 Processing task {task_id}: {zip_path}")
        print(f"📁 Output directory: {output_dir}")

        # Initialize hackathon tester
        tester = HackathonTester()

        # Process single case
        result = tester.test_single_case(str(zip_path))

        # Create Excel/CSV output in permanent location
        excel_path, csv_path = create_excel_output([result], str(output_dir))

        # Update task with results
        tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result_files": {
                "excel": str(excel_path),
                "csv": str(csv_path)
            },
            "result_data": {
                "probability_of_pathology": result["probability_of_pathology"],
                "pathology": result["pathology"],
                "processing_time": result["processing_time"],
                "status": result["status"]
            }
        })

        print(f"✅ Task {task_id} completed successfully")
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

async def process_dicom_zip(task_id: str, zip_path: Path, output_dir: Path):
    """Async wrapper that runs heavy processing in thread pool with concurrency limit"""
    async with processing_semaphore:
        # Update task status to processing when slot becomes available
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["started_at"] = datetime.now().isoformat()

        loop = asyncio.get_event_loop()

        try:
            # Run with timeout
            await asyncio.wait_for(
                loop.run_in_executor(executor, process_dicom_zip_sync, task_id, zip_path, output_dir),
                timeout=PROCESSING_TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"⏰ Task {task_id} timed out after {PROCESSING_TIMEOUT} seconds")
            tasks[task_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error_message": f"Processing timed out after {PROCESSING_TIMEOUT} seconds"
            })

@app.post("/upload", response_model=UploadResponse)
async def upload_dicom_zip(
    file: UploadFile = File(...)
):
    """Upload DICOM ZIP file for processing"""

    # Validate file
    if not file.filename.endswith('.zip'):
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

    # Create task record
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "filename": file.filename
    }

    try:
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
                dicom_files = [f for f in file_list if f.endswith('.dcm') or ('.' not in f and len(f) > 4)]
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
        asyncio.create_task(process_dicom_zip(task_id, zip_path, task_results_dir))

        return UploadResponse(
            task_id=task_id,
            status="pending",
            message=f"File uploaded successfully. Found {len(dicom_files)} DICOM files."
        )

    except Exception as e:
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
        result_files=task_data.get("result_files")
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
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
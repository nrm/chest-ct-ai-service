"""
Немедленная генерация PNG слайсов для просмотра
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pydicom
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import tempfile
import shutil

def generate_immediate_slices(task_id: str, dicom_files: List[Path], output_dir: Path, max_slices: int = 20, slice_step: int = 1) -> Dict:
    """
    Немедленно генерирует PNG слайсы для просмотра без сегментации
    
    Args:
        task_id: ID задачи
        dicom_files: Список DICOM файлов
        output_dir: Директория для сохранения
        max_slices: Максимальное количество слайсов
        slice_step: Шаг между слайсами (каждый N-й слайс)
    
    Returns:
        Словарь с информацией о сгенерированных слайсах
    """
    print(f"📸 Генерация немедленных PNG слайсов для задачи {task_id}...")
    
    # Создаем директорию для слайсов
    slices_dir = output_dir / "immediate_slices"
    slices_dir.mkdir(exist_ok=True, parents=True)
    
    generated_slices = []
    slice_counter = 0
    
    # Обрабатываем каждый DICOM файл
    # Обрабатываем каждый файл, но берем каждый N-й слайс
    for file_idx, dicom_file in enumerate(dicom_files):
        if slice_counter >= max_slices:
            break
        
        try:
            # Загружаем DICOM файл
            ds = pydicom.dcmread(str(dicom_file))
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Проверяем, является ли это 3D массивом (множественные слайсы)
            if pixel_array.ndim == 3:
                # 3D массив - обрабатываем каждый N-й слайс
                for z in range(0, pixel_array.shape[0], slice_step):
                    if slice_counter >= max_slices:
                        break
                        
                    slice_data = pixel_array[z]
                    slice_filename = f"slice_{slice_counter:04d}.png"
                    slice_path = slices_dir / slice_filename
                    
                    # Обрабатываем слайс
                    success = _process_single_slice(
                        slice_data, ds, slice_path, slice_counter, 
                        dicom_file.name, z
                    )
                    
                    if success:
                        generated_slices.append({
                            'slice_index': slice_counter,
                            'filename': slice_filename,
                            'path': str(slice_path.relative_to(output_dir)),
                            'dicom_file': dicom_file.name,
                            'z_index': z,
                            'window_center': float(getattr(ds, 'WindowCenter', 0)),
                            'window_width': float(getattr(ds, 'WindowWidth', 400))
                        })
                        slice_counter += 1
                        print(f"  ✅ Слайс {slice_counter-1}: {slice_filename} (z={z})")
            else:
                # 2D массив - один слайс, но проверяем slice_step
                if file_idx % slice_step == 0:  # Берем каждый N-й файл
                    if slice_counter >= max_slices:
                        break
                        
                    slice_filename = f"slice_{slice_counter:04d}.png"
                    slice_path = slices_dir / slice_filename
                    
                    # Обрабатываем слайс
                    success = _process_single_slice(
                        pixel_array, ds, slice_path, slice_counter, 
                        dicom_file.name, 0
                    )
                    
                    if success:
                        generated_slices.append({
                            'slice_index': slice_counter,
                            'filename': slice_filename,
                            'path': str(slice_path.relative_to(output_dir)),
                            'dicom_file': dicom_file.name,
                            'z_index': 0,
                            'window_center': float(getattr(ds, 'WindowCenter', 0)),
                            'window_width': float(getattr(ds, 'WindowWidth', 400))
                        })
                        slice_counter += 1
                        print(f"  ✅ Слайс {slice_counter-1}: {slice_filename}")
                # else:
                #     print(f"  ⏭️ Пропускаем файл {file_idx} (slice_step={slice_step})")
            
            # Если достигли лимита, выходим из цикла по файлам
            if slice_counter >= max_slices:
                break
            
        except Exception as e:
            print(f"  ❌ Ошибка обработки файла {dicom_file.name}: {e}")
            continue
    
    # Создаем информацию о слайсах (после цикла!)
    slices_info = {
        "task_id": task_id,
        "total_dicom_files": len(dicom_files),
        "generated_slices": len(generated_slices),
        "slices_dir": str(slices_dir.relative_to(output_dir)),
        "slices": generated_slices,
        "type": "immediate",
        "generated_at": str(np.datetime64('now'))
    }
    
    # Сохраняем информацию в файл
    slices_info_path = slices_dir / "slices_info.json"
    import json
    with open(slices_info_path, 'w') as f:
        json.dump(slices_info, f, indent=2)
    
    print(f"✅ Сгенерировано {len(generated_slices)} немедленных слайсов")
    return slices_info

def _process_single_slice(slice_data: np.ndarray, ds: pydicom.Dataset, 
                         slice_path: Path, slice_idx: int, 
                         filename: str, z_idx: int) -> bool:
    """Обрабатывает один слайс и сохраняет как PNG"""
    try:
        # Применяем rescale slope и intercept если есть
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slice_data = slice_data * ds.RescaleSlope + ds.RescaleIntercept
        
        # Получаем window/level
        window_center = getattr(ds, 'WindowCenter', 0)
        window_width = getattr(ds, 'WindowWidth', 400)
        
        # Применяем window/level
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        slice_data = np.clip(slice_data, window_min, window_max)
        slice_data = (slice_data - window_min) / (window_max - window_min) * 255
        slice_data = np.clip(slice_data, 0, 255).astype(np.uint8)
        
        # Сохраняем как PNG
        plt.figure(figsize=(8, 8))
        plt.imshow(slice_data, cmap='gray', aspect='equal')
        plt.title(f'DICOM Slice {slice_idx}', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Добавляем информацию о слайсе
        info_text = f"Slice: {slice_idx}\nZ: {z_idx}\nWindow: {window_center:.0f}/{window_width:.0f}\nFile: {filename}"
        plt.figtext(0.02, 0.98, info_text, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
        
        plt.tight_layout()
        plt.savefig(slice_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка обработки слайса {slice_idx}: {e}")
        return False

def extract_dicom_files_from_zip(zip_path: Path) -> List[Path]:
    """
    Извлекает DICOM файлы из ZIP архива
    """
    import zipfile
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp(prefix=f"dicom_extract_{zip_path.stem}_"))
    dicom_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Находим DICOM файлы
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                if (file.lower().endswith('.dcm') or 
                    file.lower().endswith('.dicom') or
                    ('.' not in file and file_path.is_file())):
                    dicom_files.append(file_path)
        
        # Сортируем по имени файла
        dicom_files.sort(key=lambda x: x.name)
        
        print(f"📁 Найдено {len(dicom_files)} DICOM файлов в ZIP")
        if dicom_files:
            print(f"   Первые 5 файлов: {[f.name for f in dicom_files[:5]]}")
        return dicom_files
        
    except Exception as e:
        print(f"❌ Ошибка извлечения DICOM файлов: {e}")
        return []

def cleanup_temp_dir(temp_dir: Path):
    """Очищает временную директорию"""
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"⚠️  Не удалось очистить временную директорию {temp_dir}: {e}")

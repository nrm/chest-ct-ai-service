"""
Интеграция с ct_lung.py сегментатором для сегментации костей
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Добавляем путь к ct_lung.py
CT_LUNG_PATH = Path(__file__).parent.parent / "segment_and_viz_1"
sys.path.insert(0, str(CT_LUNG_PATH))

try:
    from ct_lung import segment_bones, _safe_binary_opening, _ball_structure, _voxel_volume_mm3, _remove_small, _keep_largest
    CT_LUNG_AVAILABLE = True
    print("✅ ct_lung.py сегментатор доступен")
except ImportError as e:
    CT_LUNG_AVAILABLE = False
    print(f"⚠️ ct_lung.py сегментатор недоступен: {e}")

def segment_bones_with_ct_lung(volume: np.ndarray, spacing_zyx: Tuple[float, float, float], body_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Сегментация костей с использованием ct_lung.py
    
    Args:
        volume: 3D массив HU значений
        spacing_zyx: Размеры вокселей (z, y, x)
        body_mask: Маска тела
        
    Returns:
        Маска костей или None если сегментатор недоступен
    """
    if not CT_LUNG_AVAILABLE:
        print("⚠️ ct_lung.py недоступен, пропускаем сегментацию костей")
        return None
    
    try:
        print("🦴 Запуск сегментации костей с ct_lung.py...")
        print(f"   Volume shape: {volume.shape}")
        print(f"   Spacing: {spacing_zyx}")
        print(f"   Body mask voxels: {body_mask.sum()}")
        
        # Запускаем сегментацию костей
        bones_mask = segment_bones(volume, spacing_zyx, body_mask)
        
        print(f"✅ ct_lung.py сегментация костей завершена: {bones_mask.sum()} вокселей")
        return bones_mask
        
    except Exception as e:
        print(f"❌ Ошибка сегментации костей с ct_lung.py: {e}")
        return None

def create_enhanced_bones_mask(volume: np.ndarray, spacing_zyx: Tuple[float, float, float], 
                              body_mask: np.ndarray, existing_bones_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Создает улучшенную маску костей, комбинируя существующую и ct_lung.py
    
    Args:
        volume: 3D массив HU значений
        spacing_zyx: Размеры вокселей
        body_mask: Маска тела
        existing_bones_mask: Существующая маска костей (опционально)
        
    Returns:
        Улучшенная маска костей
    """
    # Получаем маску костей от ct_lung.py
    ct_lung_bones = segment_bones_with_ct_lung(volume, spacing_zyx, body_mask)
    
    if existing_bones_mask is not None and ct_lung_bones is not None:
        # Комбинируем обе маски
        print("🔄 Комбинирование масок костей...")
        combined_bones = np.logical_or(existing_bones_mask, ct_lung_bones).astype(np.uint8)
        print(f"   Существующая маска: {existing_bones_mask.sum()} вокселей")
        print(f"   ct_lung маска: {ct_lung_bones.sum()} вокселей")
        print(f"   Комбинированная: {combined_bones.sum()} вокселей")
        return combined_bones
    elif ct_lung_bones is not None:
        # Используем только ct_lung.py (основной случай)
        print("🦴 Используем ct_lung.py маску костей (основная сегментация)")
        return ct_lung_bones
    elif existing_bones_mask is not None:
        # Используем только существующую (fallback)
        print("🦴 Используем только существующую маску костей (fallback)")
        return existing_bones_mask
    else:
        # Создаем пустую маску (последний fallback)
        print("⚠️ Создаем пустую маску костей (fallback)")
        return np.zeros_like(body_mask, dtype=np.uint8)

def get_ct_lung_status() -> Dict[str, any]:
    """
    Возвращает статус доступности ct_lung.py сегментатора
    
    Returns:
        Словарь с информацией о статусе
    """
    return {
        "available": CT_LUNG_AVAILABLE,
        "path": str(CT_LUNG_PATH),
        "functions": {
            "segment_bones": "segment_bones" in globals() if CT_LUNG_AVAILABLE else False,
            "helper_functions": all(func in globals() for func in [
                "_safe_binary_opening", "_ball_structure", "_voxel_volume_mm3", 
                "_remove_small", "_keep_largest"
            ]) if CT_LUNG_AVAILABLE else False
        }
    }

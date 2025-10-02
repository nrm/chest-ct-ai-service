"""
Оптимизированная сегментация с интерполяцией между слайсами
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.ndimage
from scipy.interpolate import interp1d
import time

def interpolate_masks_between_slices(sparse_masks: Dict[str, np.ndarray], 
                                   target_shape: Tuple[int, int, int],
                                   slice_indices: List[int],
                                   method: str = 'linear') -> Dict[str, np.ndarray]:
    """
    Интерполирует маски между рассчитанными слайсами
    
    Args:
        sparse_masks: Словарь с масками для каждого компонента
        target_shape: Целевая форма 3D массива (z, y, x)
        slice_indices: Индексы слайсов, для которых рассчитаны маски
        method: Метод интерполяции ('linear', 'nearest', 'cubic')
    
    Returns:
        Словарь с полными 3D масками
    """
    print(f"🔄 Интерполяция масок между {len(slice_indices)} слайсами...")
    
    full_masks = {}
    z_target = target_shape[0]
    
    for comp_name, comp_masks in sparse_masks.items():
        if comp_masks is None or len(comp_masks) == 0:
            full_masks[comp_name] = np.zeros(target_shape, dtype=np.uint8)
            continue
        
        print(f"   Интерполяция {comp_name}...")
        
        # Создаем полную маску
        full_mask = np.zeros(target_shape, dtype=np.uint8)
        
        # Заполняем рассчитанные слайсы
        for i, slice_idx in enumerate(slice_indices):
            if i < len(comp_masks):
                full_mask[slice_idx] = comp_masks[i]
        
        # Интерполируем между слайсами
        for i in range(len(slice_indices) - 1):
            start_idx = slice_indices[i]
            end_idx = slice_indices[i + 1]
            start_mask = comp_masks[i]
            end_mask = comp_masks[i + 1]
            
            # Интерполируем только если есть данные в обоих слайсах
            if start_mask.any() or end_mask.any():
                interpolated = interpolate_between_two_slices(
                    start_mask, end_mask, 
                    start_idx, end_idx, 
                    method=method
                )
                
                # Заполняем промежуточные слайсы
                for j in range(start_idx + 1, end_idx):
                    if j < z_target:
                        full_mask[j] = interpolated[j - start_idx - 1]
        
        # Обрабатываем области до первого и после последнего слайса
        if slice_indices[0] > 0:
            # Копируем первый слайс в начало
            first_mask = comp_masks[0]
            for i in range(slice_indices[0]):
                full_mask[i] = first_mask
        
        if slice_indices[-1] < z_target - 1:
            # Копируем последний слайс в конец
            last_mask = comp_masks[-1]
            for i in range(slice_indices[-1] + 1, z_target):
                full_mask[i] = last_mask
        
        full_masks[comp_name] = full_mask
    
    print(f"✅ Интерполяция завершена")
    return full_masks

def interpolate_between_two_slices(start_mask: np.ndarray, 
                                 end_mask: np.ndarray,
                                 start_idx: int, 
                                 end_idx: int,
                                 method: str = 'linear') -> List[np.ndarray]:
    """
    Интерполирует маски между двумя слайсами
    """
    num_slices = end_idx - start_idx - 1
    if num_slices <= 0:
        return []
    
    interpolated = []
    
    if method == 'nearest':
        # Простое копирование ближайшего слайса
        for i in range(num_slices):
            if i < num_slices // 2:
                interpolated.append(start_mask.copy())
            else:
                interpolated.append(end_mask.copy())
    
    elif method == 'linear':
        # Линейная интерполяция
        for i in range(num_slices):
            alpha = (i + 1) / (num_slices + 1)
            interpolated_mask = (start_mask * (1 - alpha) + end_mask * alpha).astype(np.uint8)
            interpolated.append(interpolated_mask)
    
    elif method == 'morphological':
        # Морфологическая интерполяция
        for i in range(num_slices):
            alpha = (i + 1) / (num_slices + 1)
            
            # Комбинируем маски
            combined = np.logical_or(start_mask, end_mask).astype(np.uint8)
            
            # Применяем морфологические операции для сглаживания
            if combined.any():
                from skimage.morphology import binary_closing, binary_opening
                combined = binary_closing(combined, footprint=np.ones((3, 3)))
                combined = binary_opening(combined, footprint=np.ones((2, 2)))
            
            interpolated.append(combined)
    
    return interpolated

def create_sparse_segmentation(volume: np.ndarray, 
                             projector,
                             args,
                             slice_step: int = 5,
                             interpolation_method: str = 'morphological') -> Dict[str, np.ndarray]:
    """
    Создает сегментацию только для каждого N-го слайса, затем интерполирует
    
    Args:
        volume: 3D массив CT данных
        projector: Проектор для сегментации
        args: Аргументы конфигурации
        slice_step: Шаг между слайсами (каждый N-й слайс)
        interpolation_method: Метод интерполяции
    
    Returns:
        Словарь с полными 3D масками
    """
    print(f"🎯 Sparse segmentation: каждый {slice_step}-й слайс")
    
    z, y, x = volume.shape
    slice_indices = list(range(0, z, slice_step))
    
    # Если последний слайс не включен, добавляем его
    if slice_indices[-1] != z - 1:
        slice_indices.append(z - 1)
    
    print(f"   Рассчитываем маски для {len(slice_indices)} слайсов из {z}")
    
    # Рассчитываем маски только для выбранных слайсов
    sparse_masks = {}
    start_time = time.time()
    
    for comp_name in ['body', 'lungs', 'bone', 'airways', 'soft']:
        sparse_masks[comp_name] = []
    
    for i, slice_idx in enumerate(slice_indices):
        print(f"   Слайс {slice_idx}/{z-1} ({i+1}/{len(slice_indices)})")
        
        # Создаем 2D слайс
        slice_2d = volume[slice_idx]
        
        # Рассчитываем маски для этого слайса
        slice_masks = calculate_slice_masks(slice_2d, projector, args, slice_idx)
        
        # Сохраняем маски
        for comp_name, mask in slice_masks.items():
            sparse_masks[comp_name].append(mask)
    
    calculation_time = time.time() - start_time
    print(f"⚡ Расчет {len(slice_indices)} слайсов: {calculation_time:.2f}s")
    
    # Интерполируем между слайсами
    interpolation_start = time.time()
    full_masks = interpolate_masks_between_slices(
        sparse_masks, volume.shape, slice_indices, interpolation_method
    )
    interpolation_time = time.time() - interpolation_start
    print(f"🔄 Интерполяция: {interpolation_time:.2f}s")
    
    total_time = calculation_time + interpolation_time
    speedup = (z * calculation_time / len(slice_indices)) / total_time
    print(f"🚀 Общее ускорение: {speedup:.1f}x (вместо {z} слайсов - {len(slice_indices)})")
    
    return full_masks

def calculate_slice_masks(slice_2d: np.ndarray, 
                         projector, 
                         args, 
                         slice_idx: int) -> Dict[str, np.ndarray]:
    """
    Рассчитывает маски для одного 2D слайса
    """
    masks = {}
    
    try:
        # Создаем 3D массив из одного слайса для совместимости
        volume_3d = slice_2d[np.newaxis, :, :]
        
        # Используем существующие функции сегментации
        from configurable_dual_body_sementation import (
            create_big_body_mask, create_small_body_mask,
            clean_lung_artifacts_configurable, clean_airways_configurable
        )
        
        # Тело
        big_body = create_big_body_mask(volume_3d)
        small_body = create_small_body_mask(volume_3d)
        masks['body'] = small_body[0]  # Берем только 2D слайс
        
        # Легкие (упрощенная версия)
        if hasattr(projector, '_compute_lung_mask_enhanced'):
            lungs_big = projector._compute_lung_mask_enhanced(volume_3d, big_body)
            lungs_limited = np.logical_and(lungs_big, small_body).astype(np.uint8)
            lungs_final = clean_lung_artifacts_configurable(
                lungs_limited[0], volume_3d[0], 
                body_mask=small_body[0], thorax=small_body[0]
            )
            masks['lungs'] = lungs_final
        else:
            masks['lungs'] = np.zeros_like(slice_2d, dtype=np.uint8)
        
        # Кости (если требуется)
        if args.separate_bones and hasattr(projector, '_compute_bone_mask_enhanced'):
            bones_big = projector._compute_bone_mask_enhanced(volume_3d, big_body)
            masks['bone'] = bones_big[0]
        else:
            masks['bone'] = np.zeros_like(slice_2d, dtype=np.uint8)
        
        # Дыхательные пути
        if hasattr(projector, '_compute_airways_mask'):
            airways_big = projector._compute_airways_mask(volume_3d, lungs_big, big_body)
            airways_final = clean_airways_configurable(airways_big[0], lungs_final)
            masks['airways'] = airways_final
        else:
            masks['airways'] = np.zeros_like(slice_2d, dtype=np.uint8)
        
        # Мягкие ткани
        soft_mask = small_body[0].copy()
        if masks['lungs'].any():
            soft_mask[masks['lungs'] > 0] = 0
        if masks['bone'].any():
            soft_mask[masks['bone'] > 0] = 0
        if masks['airways'].any():
            soft_mask[masks['airways'] > 0] = 0
        masks['soft'] = soft_mask.astype(np.uint8)
        
    except Exception as e:
        print(f"⚠️  Ошибка расчета масок для слайса {slice_idx}: {e}")
        # Возвращаем пустые маски
        for comp_name in ['body', 'lungs', 'bone', 'airways', 'soft']:
            masks[comp_name] = np.zeros_like(slice_2d, dtype=np.uint8)
    
    return masks

def adaptive_slice_selection(volume: np.ndarray, 
                           max_slices: int = 20,
                           min_step: int = 2) -> List[int]:
    """
    Адаптивно выбирает слайсы для сегментации на основе изменений в объеме
    """
    z, y, x = volume.shape
    
    if z <= max_slices:
        return list(range(z))
    
    # Вычисляем градиент по Z-направлению для выявления изменений
    gradient = np.abs(np.diff(volume, axis=0))
    gradient_sum = np.sum(gradient, axis=(1, 2))
    
    # Находим пики изменений
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(gradient_sum, distance=min_step)
    
    # Добавляем начальный и конечный слайсы
    selected_indices = [0]
    selected_indices.extend(peaks.tolist())
    if z - 1 not in selected_indices:
        selected_indices.append(z - 1)
    
    # Ограничиваем количество слайсов
    if len(selected_indices) > max_slices:
        # Выбираем слайсы с наибольшими изменениями
        peak_values = gradient_sum[peaks]
        top_peaks = np.argsort(peak_values)[-max_slices+2:]  # +2 для начального и конечного
        selected_indices = [0] + peaks[top_peaks].tolist() + [z - 1]
        selected_indices = sorted(list(set(selected_indices)))
    
    return selected_indices

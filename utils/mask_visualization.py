"""
Утилиты для визуализации масок на DICOM слайсах
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pydicom
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
from scipy import ndimage

def _draw_thick_contours(rgba_image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int, int], thickness: int = 3):
    """
    Рисует толстые контуры маски вместо заливки
    
    Args:
        rgba_image: RGBA изображение для рисования
        mask: Бинарная маска
        color: Цвет контура (R, G, B, A)
        thickness: Толщина контура в пикселях
    """
    if not mask.any():
        return
    
    # Находим контуры маски
    contours = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1))
    contours = mask & ~contours
    
    # Создаем толстые контуры с помощью дилатации
    if thickness > 1:
        structure = ndimage.generate_binary_structure(2, 1)
        for _ in range(thickness - 1):
            contours = ndimage.binary_dilation(contours, structure=structure)
    
    # Рисуем контуры
    contour_indices = contours > 0
    rgba_image[contour_indices] = color

def load_dicom_slice(dicom_file: Path) -> Tuple[np.ndarray, Dict]:
    """Загружает DICOM слайс и возвращает массив и метаданные"""
    try:
        ds = pydicom.dcmread(str(dicom_file))
        
        # Получаем пиксельные данные
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Применяем rescale slope и intercept если есть
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Получаем метаданные
        metadata = {
            'window_center': getattr(ds, 'WindowCenter', 0),
            'window_width': getattr(ds, 'WindowWidth', 400),
            'photometric_interpretation': getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2'),
            'rows': ds.Rows,
            'columns': ds.Columns,
            'slice_thickness': getattr(ds, 'SliceThickness', 1.0),
            'pixel_spacing': getattr(ds, 'PixelSpacing', [1.0, 1.0])
        }
        
        return pixel_array, metadata
        
    except Exception as e:
        print(f"Ошибка загрузки DICOM файла {dicom_file}: {e}")
        return None, {}

def apply_window_level(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Применяет window/level к изображению"""
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    # Ограничиваем значения окном
    image = np.clip(image, window_min, window_max)
    
    # Нормализуем к 0-255
    image = (image - window_min) / (window_max - window_min) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image

def create_mask_overlay(base_image: np.ndarray, mask: np.ndarray, 
                       color: Tuple[int, int, int, int] = (0, 255, 0, 128),
                       alpha: float = 0.5) -> np.ndarray:
    """Создает наложение маски на базовое изображение"""
    
    # Создаем RGB изображение
    if len(base_image.shape) == 2:
        rgb_image = np.stack([base_image] * 3, axis=-1)
    else:
        rgb_image = base_image.copy()
    
    # Создаем RGBA изображение
    rgba_image = np.concatenate([rgb_image, np.full((*rgb_image.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
    
    # Применяем маску
    mask_indices = mask > 0
    rgba_image[mask_indices] = color
    
    return rgba_image

def create_multi_mask_overlay(base_image: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Создает наложение нескольких масок с разными цветами"""
    
    # Цвета для разных компонент
    colors = {
        'body': (128, 128, 128, 100),      # Серый
        'lungs': (0, 255, 255, 150),       # Голубой
        'bone': (255, 255, 0, 200),        # Желтый
        'spine': (255, 165, 0, 200),       # Оранжевый
        'ribs': (255, 255, 100, 150),      # Светло-желтый
        'airways': (0, 255, 0, 200),       # Зеленый
        'soft': (255, 0, 255, 100),        # Пурпурный
    }
    
    # Создаем RGB изображение
    if len(base_image.shape) == 2:
        rgb_image = np.stack([base_image] * 3, axis=-1)
    else:
        rgb_image = base_image.copy()
    
    # Создаем RGBA изображение
    rgba_image = np.concatenate([rgb_image, np.full((*rgb_image.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
    
    # Применяем маски в порядке приоритета
    for comp_name, mask in masks.items():
        if mask is not None and mask.any():
            print(f"      Применяем маску {comp_name}: shape={mask.shape}, voxels={mask.sum()}")
            
            # Если маска 3D, берем срез по Z (slice_index)
            if len(mask.shape) == 3:
                # Для 3D масок нужно передать slice_index, но пока берем первый срез
                mask_2d = mask[0] if mask.shape[0] > 0 else mask[:, :, 0]
            else:
                mask_2d = mask
            
            # Проверяем, что размеры совпадают
            if mask_2d.shape[:2] != base_image.shape[:2]:
                print(f"      ⚠️ Размеры не совпадают: mask={mask_2d.shape}, image={base_image.shape}")
                continue
                
            color = colors.get(comp_name, (255, 0, 0, 150))  # Красный по умолчанию
            # Рисуем толстые контуры вместо заливки
            _draw_thick_contours(rgba_image, mask_2d, color, thickness=3)
        else:
            print(f"      Пропускаем пустую маску {comp_name}")
    
    return rgba_image

def create_multi_mask_overlay_with_slice(base_image: np.ndarray, masks: Dict[str, np.ndarray], slice_index: int) -> np.ndarray:
    """Создает наложение нескольких масок с разными цветами для конкретного среза"""
    
    # Цвета для разных компонент
    colors = {
        'body': (128, 128, 128, 100),      # Серый
        'lungs': (0, 255, 255, 150),       # Голубой
        'bone': (255, 255, 0, 200),        # Желтый
        'spine': (255, 165, 0, 200),       # Оранжевый
        'ribs': (255, 255, 100, 150),      # Светло-желтый
        'airways': (0, 255, 0, 200),       # Зеленый
        'soft': (255, 0, 255, 100),        # Пурпурный
    }
    
    # Создаем RGB изображение
    if len(base_image.shape) == 2:
        rgb_image = np.stack([base_image] * 3, axis=-1)
    else:
        rgb_image = base_image.copy()
    
    # Создаем RGBA изображение
    rgba_image = np.concatenate([rgb_image, np.full((*rgb_image.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
    
    # Применяем маски в порядке приоритета
    for comp_name, mask in masks.items():
        if mask is not None and mask.any():
            print(f"      Применяем маску {comp_name}: shape={mask.shape}, voxels={mask.sum()}")
            
            # Если маска 3D, берем срез по Z (slice_index)
            if len(mask.shape) == 3:
                # slice_index - это индекс DICOM файла, используем его как Z-координату
                # Но нужно убедиться, что индекс не превышает размер маски
                z_idx = min(slice_index, mask.shape[0] - 1)
                mask_2d = mask[z_idx]
                print(f"      Используем срез {z_idx} из 3D маски {comp_name} (shape={mask.shape})")
            else:
                mask_2d = mask
                print(f"      Используем 2D маску {comp_name} (shape={mask.shape})")
            
            # Проверяем, что размеры совпадают
            if mask_2d.shape[:2] != base_image.shape[:2]:
                print(f"      ⚠️ Размеры не совпадают: mask={mask_2d.shape}, image={base_image.shape}")
                continue
                
            color = colors.get(comp_name, (255, 0, 0, 150))  # Красный по умолчанию
            # Рисуем толстые контуры вместо заливки
            _draw_thick_contours(rgba_image, mask_2d, color, thickness=3)
        else:
            print(f"      Пропускаем пустую маску {comp_name}")
    
    return rgba_image

def create_annotated_slice(dicom_file: Path, masks: Dict[str, np.ndarray], 
                          slice_index: int, output_path: Path,
                          show_legend: bool = True) -> bool:
    """Создает аннотированный слайс с наложением масок"""
    
    try:
        print(f"    Создание слайса {slice_index} с масками...")
        print(f"    Доступные маски: {list(masks.keys())}")
        
        # Загружаем DICOM слайс
        pixel_array, metadata = load_dicom_slice(dicom_file)
        if pixel_array is None:
            print(f"    ❌ Не удалось загрузить DICOM файл {dicom_file}")
            return False
        
        print(f"    DICOM загружен: shape={pixel_array.shape}")
        
        # Применяем window/level
        window_center = metadata.get('window_center', 0)
        window_width = metadata.get('window_width', 400)
        processed_image = apply_window_level(pixel_array, window_center, window_width)
        
        # Создаем наложение масок
        print(f"    Создание наложения масок для slice_index={slice_index}...")
        overlay_image = create_multi_mask_overlay_with_slice(processed_image, masks, slice_index)
        print(f"    Наложение создано: shape={overlay_image.shape}")
        
        # Создаем фигуру
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Отображаем изображение
        ax.imshow(overlay_image, cmap='gray', aspect='equal')
        ax.set_title(f'DICOM Slice {slice_index} with Segmentation Masks', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Добавляем легенду
        if show_legend:
            legend_elements = []
            colors = {
                'body': (128, 128, 128),      # Серый
                'lungs': (0, 255, 255),       # Голубой
                'bone': (255, 255, 0),        # Желтый
                'spine': (255, 165, 0),       # Оранжевый
                'ribs': (255, 255, 100),      # Светло-желтый
                'airways': (0, 255, 0),       # Зеленый
                'soft': (255, 0, 255),        # Пурпурный
            }
            
            for comp_name, mask in masks.items():
                if mask is not None and mask.any():
                    color = colors.get(comp_name, (255, 0, 0))
                    legend_elements.append(
                        patches.Patch(color=[c/255.0 for c in color], 
                                    label=f'{comp_name.replace("_", " ").title()} ({mask.sum():,} voxels)')
                    )
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', 
                         bbox_to_anchor=(1.0, 1.0), fontsize=10)
        
        # Добавляем информацию о слайсе
        info_text = f"Slice: {slice_index}\nWindow: {window_center:.0f}/{window_width:.0f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        # Сохраняем
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Ошибка создания аннотированного слайса: {e}")
        return False

def create_slice_with_bbox(dicom_file: Path, bbox: Tuple[float, float, float, float], 
                          output_path: Path) -> bool:
    """Создает слайс с выделенным bounding box"""
    
    try:
        # Загружаем DICOM слайс
        pixel_array, metadata = load_dicom_slice(dicom_file)
        if pixel_array is None:
            return False
        
        # Применяем window/level
        window_center = metadata.get('window_center', 0)
        window_width = metadata.get('window_width', 400)
        processed_image = apply_window_level(pixel_array, window_center, window_width)
        
        # Создаем фигуру
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Отображаем изображение
        ax.imshow(processed_image, cmap='gray', aspect='equal')
        
        # Добавляем bounding box
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Добавляем текст
            ax.text(x1, y1-10, 'Pathology Region', color='red', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title('DICOM Slice with Pathology Bounding Box', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Сохраняем
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Ошибка создания слайса с bbox: {e}")
        return False

def generate_mask_slices_for_task(task_id: str, dicom_files: List[Path], 
                                 masks: Dict[str, np.ndarray], output_dir: Path,
                                 slice_step: int = 3) -> List[Dict]:
    """Генерирует PNG слайсы с масками для задачи (каждый N-й слайс)"""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    generated_slices = []
    
    # Выбираем слайсы для генерации (каждый N-й)
    selected_indices = list(range(0, len(dicom_files), slice_step))
    
    print(f"Генерация {len(selected_indices)} слайсов с масками для задачи {task_id}")
    print(f"  Всего DICOM файлов: {len(dicom_files)}")
    print(f"  Форма масок: {list(masks.values())[0].shape if masks else 'N/A'}")
    
    # Проверка соответствия размеров
    if masks:
        first_mask = list(masks.values())[0]
        if hasattr(first_mask, 'shape') and len(first_mask.shape) == 3:
            mask_z = first_mask.shape[0]
            dicom_count = len(dicom_files)
            if mask_z != dicom_count:
                print(f"  ⚠️ НЕСООТВЕТСТВИЕ: маска имеет {mask_z} слоёв, но DICOM файлов {dicom_count}")
                print(f"  ⚠️ Это может привести к неправильному наложению масок!")
            else:
                print(f"  ✅ Размеры совпадают: {mask_z} слоёв")
    
    for i, slice_idx in enumerate(selected_indices):
        dicom_file = dicom_files[slice_idx]
        
        # Создаем выходной файл
        slice_output_path = output_dir / f"slice_{slice_idx:04d}_with_masks.png"
        
        # Генерируем слайс с масками
        # slice_idx - это индекс в отсортированном списке DICOM файлов
        # Он соответствует Z-координате в 3D маске
        success = create_annotated_slice(dicom_file, masks, slice_idx, slice_output_path)
        
        if success:
            generated_slices.append({
                'slice_index': slice_idx,
                'filename': slice_output_path.name,
                'path': str(slice_output_path.relative_to(output_dir.parent)),
                'dicom_file': dicom_file.name
            })
            print(f"  ✅ Слайс {slice_idx}: {slice_output_path.name}")
        else:
            print(f"  ❌ Ошибка генерации слайса {slice_idx}")
    
    return generated_slices

def create_comparison_view(dicom_file: Path, masks: Dict[str, np.ndarray], 
                          slice_index: int, output_path: Path) -> bool:
    """Создает сравнительный вид: оригинал, маски, наложение"""
    
    try:
        # Загружаем DICOM слайс
        pixel_array, metadata = load_dicom_slice(dicom_file)
        if pixel_array is None:
            return False
        
        # Применяем window/level
        window_center = metadata.get('window_center', 0)
        window_width = metadata.get('window_width', 400)
        processed_image = apply_window_level(pixel_array, window_center, window_width)
        
        # Создаем фигуру с тремя подграфиками
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Оригинальный слайс
        axes[0].imshow(processed_image, cmap='gray', aspect='equal')
        axes[0].set_title('Original DICOM Slice', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Только маски
        mask_overlay = create_multi_mask_overlay(np.zeros_like(processed_image), masks)
        axes[1].imshow(mask_overlay, aspect='equal')
        axes[1].set_title('Segmentation Masks Only', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 3. Наложение
        overlay_image = create_multi_mask_overlay(processed_image, masks)
        axes[2].imshow(overlay_image, aspect='equal')
        axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Общий заголовок
        fig.suptitle(f'DICOM Slice {slice_index} - Segmentation Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Сохраняем
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Ошибка создания сравнительного вида: {e}")
        return False

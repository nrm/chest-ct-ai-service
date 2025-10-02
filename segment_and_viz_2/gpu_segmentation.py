#!/usr/bin/env python3
"""
GPU-оптимизированная сегментация для RadiAssist
Использует CuPy для ускорения вычислений на GPU
"""

import sys
import os
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, Optional, Tuple, Any
import argparse

# Настройка окружения для GPU
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🔥 GPU (CuPy) доступен")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU (CuPy) недоступен, используем CPU")

try:
    from ct_mip_visualization import CTVisualizer, SegmentationHelper
    from configurable_dual_body_sementation import (
        create_big_body_mask, create_small_body_mask, create_convex_hull_body,
        clean_bone_mask_configurable, clean_lung_artifacts_configurable,
        clean_airways_configurable, separate_bones_configurable,
        build_thoracic_container_from_body_and_bone, remove_small_components
    )
    SEGMENTATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Segmentation modules not available: {e}")
    SEGMENTATION_AVAILABLE = False


class GPUSegmentationProcessor:
    """GPU-оптимизированный процессор сегментации"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device = cp if self.use_gpu else np
        
    def to_gpu(self, array: np.ndarray) -> Any:
        """Переводит массив на GPU"""
        if self.use_gpu:
            return cp.asarray(array)
        return array
    
    def to_cpu(self, array: Any) -> np.ndarray:
        """Переводит массив обратно на CPU"""
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()
        return array
    
    def gpu_binary_operations(self, mask1: Any, mask2: Any, operation: str) -> Any:
        """GPU-оптимизированные бинарные операции"""
        if not self.use_gpu:
            if operation == 'and':
                return mask1 & mask2
            elif operation == 'or':
                return mask1 | mask2
            elif operation == 'not':
                return ~mask1
            return mask1
        
        if operation == 'and':
            return self.device.logical_and(mask1, mask2)
        elif operation == 'or':
            return self.device.logical_or(mask1, mask2)
        elif operation == 'not':
            return self.device.logical_not(mask1)
        return mask1
    
    def gpu_morphology(self, mask: Any, operation: str, footprint_size: int = 3) -> Any:
        """GPU-оптимизированные морфологические операции"""
        if not self.use_gpu:
            from skimage.morphology import binary_opening, binary_closing, binary_erosion, binary_dilation
            footprint = np.ones((footprint_size, footprint_size, footprint_size))
            
            if operation == 'opening':
                return binary_opening(mask, footprint=footprint)
            elif operation == 'closing':
                return binary_closing(mask, footprint=footprint)
            elif operation == 'erosion':
                return binary_erosion(mask, footprint=footprint)
            elif operation == 'dilation':
                return binary_dilation(mask, footprint=footprint)
            return mask
        
        # GPU-версия морфологических операций через CuPy
        # Для простоты используем CPU для морфологии, так как CuPy не имеет ndimage
        # Но ускоряем другие операции
        cpu_mask = self.to_cpu(mask)
        from skimage.morphology import binary_opening, binary_closing, binary_erosion, binary_dilation
        footprint = np.ones((footprint_size, footprint_size, footprint_size))
        
        if operation == 'opening':
            result = binary_opening(cpu_mask, footprint=footprint)
        elif operation == 'closing':
            result = binary_closing(cpu_mask, footprint=footprint)
        elif operation == 'erosion':
            result = binary_erosion(cpu_mask, footprint=footprint)
        elif operation == 'dilation':
            result = binary_dilation(cpu_mask, footprint=footprint)
        else:
            result = cpu_mask
        
        return self.to_gpu(result)
    
    def gpu_threshold(self, volume: Any, threshold: float) -> Any:
        """GPU-оптимизированное пороговое значение"""
        return volume > threshold
    
    def gpu_connected_components(self, mask: Any) -> Tuple[Any, int]:
        """GPU-оптимизированное нахождение связанных компонент"""
        if not self.use_gpu:
            from scipy import ndimage
            labeled, num_components = ndimage.label(mask)
            return labeled, num_components
        
        # GPU-версия через CuPy
        labeled = cp.asarray(cp.ndimage.label(mask)[0])
        num_components = int(cp.max(labeled))
        return labeled, num_components
    
    def gpu_largest_components(self, mask: Any, n_components: int = 1) -> Any:
        """GPU-оптимизированное выделение крупнейших компонент"""
        if not self.use_gpu:
            return SegmentationHelper.get_largest_components(mask, n_components)
        
        # GPU-версия
        labeled, num_components = self.gpu_connected_components(mask)
        
        if num_components == 0:
            return mask
        
        # Находим размеры компонент
        component_sizes = cp.bincount(labeled.ravel())[1:]  # Игнорируем фон (0)
        
        # Находим индексы крупнейших компонент
        if n_components >= num_components:
            largest_indices = cp.arange(1, num_components + 1)
        else:
            largest_indices = cp.argsort(component_sizes)[-n_components:] + 1
        
        # Создаем маску только для крупнейших компонент
        result_mask = cp.zeros_like(mask, dtype=cp.uint8)
        for idx in largest_indices:
            result_mask[labeled == idx] = 1
        
        return result_mask
    
    def gpu_convex_hull_2d(self, mask_slice: Any) -> Any:
        """GPU-оптимизированная выпуклая оболочка для 2D среза"""
        if not self.use_gpu:
            from skimage.morphology import convex_hull_image
            return convex_hull_image(mask_slice.astype(bool))
        
        # GPU-версия выпуклой оболочки
        # Для простоты используем CPU для этой операции, так как она сложная для GPU
        cpu_slice = self.to_cpu(mask_slice)
        from skimage.morphology import convex_hull_image
        hull = convex_hull_image(cpu_slice.astype(bool))
        return self.to_gpu(hull.astype(np.uint8))
    
    def create_gpu_configurable_masks(self, volume: np.ndarray, projector, args, case_name=None) -> Dict[str, np.ndarray]:
        """GPU-оптимизированное создание конфигурируемых масок"""
        
        print(f"🔥 GPU сегментация: {'включена' if self.use_gpu else 'отключена'}")
        
        # Переводим объем на GPU
        gpu_volume = self.to_gpu(volume)
        masks = {}
        
        # 1. Создание масок тела (CPU, так как использует сложные алгоритмы)
        print("     Создание масок тела...")
        big_body = create_big_body_mask(volume)  # CPU
        small_body = create_small_body_mask(volume)  # CPU
        convex_body = create_convex_hull_body(small_body)  # CPU
        
        # Переводим на GPU для дальнейших операций
        gpu_big_body = self.to_gpu(big_body)
        gpu_small_body = self.to_gpu(small_body)
        gpu_convex_body = self.to_gpu(convex_body)
        
        masks['body'] = small_body  # Сохраняем CPU версию
        
        # 2. Сегментация костей (если требуется)
        bone_final = None
        if args.separate_bones:
            print("     Сегментация костей...")
            # Используем CPU для сложных алгоритмов костей
            bones_big = projector._compute_bone_mask_enhanced(volume, big_body)
            bone_final = clean_bone_mask_configurable(
                (bones_big & convex_body).astype(np.uint8), volume, small_body
            )
            masks['bone'] = bone_final
        
        # 3. Контейнер грудной клетки
        thorax = build_thoracic_container_from_body_and_bone(convex_body, bone_final) if args.separate_bones else convex_body
        gpu_thorax = self.to_gpu(thorax)
        
        # 4. Сегментация легких (GPU-оптимизированная)
        print("     Сегментация легких...")
        lungs_big = projector._compute_lung_mask_enhanced(volume, big_body)  # CPU
        gpu_lungs_big = self.to_gpu(lungs_big)
        
        # GPU операции для легких
        gpu_lungs_limited = self.gpu_binary_operations(gpu_lungs_big, gpu_thorax, 'and')
        gpu_lungs_limited = gpu_lungs_limited.astype(self.device.uint8)
        
        # Переводим обратно на CPU для сложной очистки
        lungs_limited = self.to_cpu(gpu_lungs_limited)
        lungs_final = clean_lung_artifacts_configurable(
            lungs_limited, volume, body_mask=small_body, thorax=thorax
        )
        
        # Анти-спина: вырезаем заднюю 8% «скорлупу»
        posterior_cut = int(lungs_final.shape[1] * 0.08)
        lungs_final[:, :posterior_cut, :] = 0
        
        masks['lungs'] = lungs_final
        
        # 5. Дыхательные пути
        print("     Сегментация дыхательных путей...")
        airways_big = projector._compute_airways_mask(volume, lungs_big, big_body)  # CPU
        gpu_airways_big = self.to_gpu(airways_big)
        
        gpu_airways_limited = self.gpu_binary_operations(gpu_airways_big, gpu_convex_body, 'and')
        gpu_airways_limited = gpu_airways_limited.astype(self.device.uint8)
        
        airways_limited = self.to_cpu(gpu_airways_limited)
        airways_final = clean_airways_configurable(airways_limited, lungs_final)
        
        # 6. Мягкие ткани (GPU-оптимизированная)
        print("     Сегментация мягких тканей...")
        gpu_soft_mask = gpu_small_body.copy()
        gpu_lungs_final = self.to_gpu(lungs_final)
        gpu_airways_final = self.to_gpu(airways_final)
        
        # GPU операции для мягких тканей
        gpu_soft_mask = self.gpu_binary_operations(gpu_soft_mask, gpu_lungs_final, 'and')
        gpu_soft_mask = self.gpu_binary_operations(gpu_soft_mask, gpu_lungs_final, 'not')
        
        if args.separate_bones and 'bone' in masks:
            gpu_bone_final = self.to_gpu(masks['bone'])
            gpu_soft_mask = self.gpu_binary_operations(gpu_soft_mask, gpu_bone_final, 'and')
            gpu_soft_mask = self.gpu_binary_operations(gpu_soft_mask, gpu_bone_final, 'not')
        
        gpu_soft_mask = self.gpu_binary_operations(gpu_soft_mask, gpu_airways_final, 'and')
        gpu_soft_mask = self.gpu_binary_operations(gpu_soft_mask, gpu_airways_final, 'not')
        
        # Морфологическая очистка на GPU
        gpu_soft_mask = self.gpu_morphology(gpu_soft_mask, 'opening', 3)
        gpu_soft_mask = gpu_soft_mask.astype(self.device.uint8)
        
        # Переводим на CPU для удаления мелких компонент
        soft_mask = self.to_cpu(gpu_soft_mask)
        total_vox = int(small_body.sum())
        min_vox = 800 if total_vox < 5_000_000 else 3000
        soft_mask = remove_small_components(soft_mask, min_voxels=min_vox)
        masks['soft'] = soft_mask
        
        # 7. Деление костей (если требуется)
        if args.separate_bones and args.divide_bones and bone_final is not None:
            print("     Разделение костей...")
            spine_mask, ribs_mask = separate_bones_configurable(bone_final)
            masks['spine'] = spine_mask
            masks['ribs'] = ribs_mask
        
        print(f"✅ GPU сегментация завершена")
        return masks


def create_gpu_configurable_masks(volume: np.ndarray, projector, args, case_name=None, use_gpu: bool = True) -> Dict[str, np.ndarray]:
    """
    Создаёт конфигурируемые маски с GPU-ускорением
    
    Args:
        volume: 3D массив CT данных
        projector: Проектор для сегментации
        args: Аргументы конфигурации
        case_name: Имя кейса
        use_gpu: Использовать ли GPU ускорение
    
    Returns:
        Словарь с масками компонент
    """
    processor = GPUSegmentationProcessor(use_gpu=use_gpu)
    return processor.create_gpu_configurable_masks(volume, projector, args, case_name)


def process_single_case_gpu(case_name, data_dir, output_dir, args, use_gpu: bool = True):
    """GPU-оптимизированная обработка одного кейса"""
    
    print(f"🔥 GPU обработка кейса: {case_name}")
    print(f"   GPU: {'включен' if use_gpu else 'отключен'}")
    
    # Загрузка данных
    print(f"1. Загрузка данных для {case_name}...")
    visualizer = CTVisualizer(data_dir, output_dir)
    visualizer.load_data()
    volume = visualizer.volume
    
    print(f"   Диапазон HU: [{volume.min():.0f}, {volume.max():.0f}]")
    print(f"   Размер: {volume.shape} ({volume.nbytes / 1024**3:.2f} GB)")
    
    # GPU сегментация
    print("\n2. GPU сегментация...")
    start_time = time.time()
    
    masks = create_gpu_configurable_masks(volume, visualizer.projector, args, case_name, use_gpu)
    
    segmentation_time = time.time() - start_time
    print(f"   Время сегментации: {segmentation_time:.2f} сек")
    
    # Анализ результатов
    print("\n3. Анализ результатов...")
    analyze_gpu_results(volume, masks, args, case_name, segmentation_time)
    
    # Создание визуализаций
    print("\n4. Создание визуализаций...")
    create_gpu_visualizations(volume, masks, visualizer.metadata, output_dir, args, case_name)
    
    print(f"\n🔥 GPU ОБРАБОТКА {case_name} ЗАВЕРШЕНА!")
    return True


def analyze_gpu_results(volume: np.ndarray, masks: Dict[str, np.ndarray], args, case_name: str, segmentation_time: float):
    """Анализ результатов GPU сегментации"""
    
    print(f"   📊 Анализ результатов для {case_name}:")
    print(f"   ⏱️  Время сегментации: {segmentation_time:.2f} сек")
    
    total_voxels = volume.size
    print(f"   📏 Общий объем: {total_voxels:,} вокселей")
    
    for comp_name, mask in masks.items():
        if mask is not None and mask.any():
            voxel_count = mask.sum()
            percentage = (voxel_count / total_voxels) * 100
            print(f"   {comp_name:>12}: {voxel_count:>8,} вокселей ({percentage:>5.1f}%)")
        else:
            print(f"   {comp_name:>12}: {'не найдено':>8}")


def create_gpu_visualizations(volume: np.ndarray, masks: Dict[str, np.ndarray], metadata: Dict, 
                             output_dir: Path, args, case_name: str):
    """Создание визуализаций для GPU сегментации"""
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Создаем MIP визуализации для каждого компонента
        views = {
            'Аксиальная': 0,
            'Корональная': 1,
            'Сагиттальная': 2,
        }
        
        available_components = [comp for comp, mask in masks.items() if mask is not None and mask.any()]
        n_components = len(available_components) + 1  # +1 для исходного изображения
        
        fig, axes = plt.subplots(n_components, 3, figsize=(18, 6 * n_components))
        fig.suptitle(f'🔥 GPU СЕГМЕНТАЦИЯ: {case_name}', 
                    fontsize=16, fontweight='bold', color='darkred')
        
        # Исходный том
        for col, (view_name, axis) in enumerate(views.items()):
            if n_components == 1:
                ax = axes[col] if n_components == 1 else axes[0, col]
            else:
                ax = axes[0, col]
            
            # Создаем MIP
            if axis == 0:  # Аксиальная
                mip_img = np.max(volume, axis=axis)
            elif axis == 1:  # Корональная
                mip_img = np.max(volume, axis=axis)
            else:  # Сагиттальная
                mip_img = np.max(volume, axis=axis)
            
            # Нормализация для отображения
            mip_img = np.clip(mip_img, -1000, 500)
            mip_img = (mip_img - mip_img.min()) / (mip_img.max() - mip_img.min())
            
            ax.imshow(mip_img, cmap='gray', aspect='auto')
            ax.set_title(f'{view_name} (исходный)', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Компонентные MIP
        for row, component in enumerate(available_components, 1):
            mask = masks[component]
            
            for col, (view_name, axis) in enumerate(views.items()):
                if n_components == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
                
                # Создаем MIP маски
                if axis == 0:  # Аксиальная
                    mip_mask = np.max(mask, axis=axis)
                elif axis == 1:  # Корональная
                    mip_mask = np.max(mask, axis=axis)
                else:  # Сагиттальная
                    mip_mask = np.max(mask, axis=axis)
                
                # Нормализация
                if mip_mask.max() > 0:
                    mip_mask = mip_mask.astype(np.float32) / mip_mask.max()
                
                ax.imshow(mip_mask, cmap='hot', aspect='auto', alpha=0.7)
                
                voxel_count = mask.sum()
                title = f'{view_name} ({component})\n{voxel_count:,} вокселей 🔥'
                ax.set_title(title, fontsize=11, color='darkred', fontweight='bold')
                ax.axis('off')
        
        plt.tight_layout()
        out = output_dir / f"{case_name}_gpu_segmentation.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   🔥 GPU визуализация: {out.name}")
        
    except Exception as e:
        print(f"   ⚠️  Ошибка создания визуализации: {e}")


if __name__ == '__main__':
    # Тестирование GPU сегментации
    parser = argparse.ArgumentParser(description='GPU сегментация CT')
    parser.add_argument('--data_dir', type=str, required=True, help='Путь к папке с кейсами')
    parser.add_argument('--case', type=str, default=None, help='Название кейса')
    parser.add_argument('--separate_bones', action='store_true', help='Выделять кости')
    parser.add_argument('--divide_bones', action='store_true', help='Разделять кости')
    parser.add_argument('--output_dir', type=str, default=None, help='Папка для результатов')
    parser.add_argument('--no_gpu', action='store_true', help='Отключить GPU')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'gpu_visualizations'
    
    output_dir.mkdir(exist_ok=True)
    
    use_gpu = not args.no_gpu and GPU_AVAILABLE
    
    if args.case:
        case_dir = data_root / args.case
        if case_dir.exists():
            success = process_single_case_gpu(args.case, case_dir, output_dir, args, use_gpu)
            print(f"Результат: {'Успех' if success else 'Ошибка'}")
        else:
            print(f"Кейс не найден: {case_dir}")
    else:
        print("Укажите кейс с --case")

#!/usr/bin/env python3
"""
Оптимизированная GPU сегментация для RadiAssist
Фокус на реальном ускорении через минимизацию CPU-GPU переключений
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


class OptimizedGPUSegmentationProcessor:
    """Оптимизированный GPU процессор сегментации с минимизацией CPU-GPU переключений"""
    
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
    
    def gpu_batch_operations(self, operations: list) -> Any:
        """Выполняет пакет операций на GPU без переключений"""
        if not self.use_gpu:
            # CPU fallback
            result = operations[0]
            for op in operations[1:]:
                if op['type'] == 'and':
                    result = result & op['data']
                elif op['type'] == 'or':
                    result = result | op['data']
                elif op['type'] == 'not':
                    result = ~result
                elif op['type'] == 'threshold':
                    result = result > op['threshold']
            return result
        
        # GPU batch processing
        result = operations[0]
        for op in operations[1:]:
            if op['type'] == 'and':
                result = self.device.logical_and(result, op['data'])
            elif op['type'] == 'or':
                result = self.device.logical_or(result, op['data'])
            elif op['type'] == 'not':
                result = self.device.logical_not(result)
            elif op['type'] == 'threshold':
                result = result > op['threshold']
        return result
    
    def create_optimized_gpu_masks(self, volume: np.ndarray, projector, args, case_name=None) -> Dict[str, np.ndarray]:
        """Оптимизированное создание масок с минимизацией CPU-GPU переключений"""
        
        print(f"🔥 Оптимизированная GPU сегментация: {'включена' if self.use_gpu else 'отключена'}")
        
        # 1. Подготовка данных - все на CPU сначала
        print("     Подготовка данных...")
        big_body = create_big_body_mask(volume)
        small_body = create_small_body_mask(volume)
        convex_body = create_convex_hull_body(small_body)
        
        masks = {'body': small_body}
        
        # 2. Переводим основные данные на GPU ОДИН РАЗ
        if self.use_gpu:
            print("     Передача данных на GPU...")
            gpu_volume = self.to_gpu(volume)
            gpu_big_body = self.to_gpu(big_body)
            gpu_small_body = self.to_gpu(small_body)
            gpu_convex_body = self.to_gpu(convex_body)
        else:
            gpu_volume = volume
            gpu_big_body = big_body
            gpu_small_body = small_body
            gpu_convex_body = convex_body
        
        # 3. Сегментация костей (если требуется) - CPU для сложных алгоритмов
        bone_final = None
        if args.separate_bones:
            print("     Сегментация костей...")
            bones_big = projector._compute_bone_mask_enhanced(volume, big_body)
            bone_final = clean_bone_mask_configurable(
                (bones_big & convex_body).astype(np.uint8), volume, small_body
            )
            masks['bone'] = bone_final
            if self.use_gpu:
                gpu_bone_final = self.to_gpu(bone_final)
            else:
                gpu_bone_final = bone_final
        else:
            gpu_bone_final = None
        
        # 4. Контейнер грудной клетки
        thorax = build_thoracic_container_from_body_and_bone(convex_body, bone_final) if args.separate_bones else convex_body
        print(f"     Thorax: shape={thorax.shape}, voxels={thorax.sum()}")
        if self.use_gpu:
            gpu_thorax = self.to_gpu(thorax)
        else:
            gpu_thorax = thorax
        
        # 5. Сегментация легких - GPU-оптимизированная
        print("     Сегментация легких...")
        lungs_big = projector._compute_lung_mask_enhanced(volume, big_body)
        print(f"     Lungs big: shape={lungs_big.shape}, voxels={lungs_big.sum()}")
        
        if self.use_gpu:
            gpu_lungs_big = self.to_gpu(lungs_big)
            # Все операции на GPU
            gpu_lungs_limited = self.device.logical_and(gpu_lungs_big, gpu_thorax)
            gpu_lungs_limited = gpu_lungs_limited.astype(self.device.uint8)
            
            # Переводим обратно на CPU для сложной очистки
            lungs_limited = self.to_cpu(gpu_lungs_limited)
        else:
            lungs_limited = (lungs_big & thorax).astype(np.uint8)
        
        # Сложная очистка на CPU
        print(f"     Lungs limited before cleaning: shape={lungs_limited.shape}, voxels={lungs_limited.sum()}")
        lungs_final = clean_lung_artifacts_configurable(
            lungs_limited, volume, body_mask=small_body, thorax=thorax
        )
        print(f"     Lungs after cleaning: shape={lungs_final.shape}, voxels={lungs_final.sum()}")
        
        # Анти-спина
        posterior_cut = int(lungs_final.shape[1] * 0.08)
        lungs_final[:, :posterior_cut, :] = 0
        print(f"     Lungs after posterior cut: shape={lungs_final.shape}, voxels={lungs_final.sum()}")
        masks['lungs'] = lungs_final
        print(f"     Lungs mask: shape={lungs_final.shape}, voxels={lungs_final.sum()}")
        
        # 6. Дыхательные пути
        # print("     Сегментация дыхательных путей...")
        # airways_big = projector._compute_airways_mask(volume, lungs_big, big_body)
        # print(f"     Airways big: shape={airways_big.shape}, voxels={airways_big.sum()}")
        # 
        # if self.use_gpu:
        #     gpu_airways_big = self.to_gpu(airways_big)
        #     gpu_airways_limited = self.device.logical_and(gpu_airways_big, gpu_convex_body)
        #     gpu_airways_limited = gpu_airways_limited.astype(self.device.uint8)
        #     airways_limited = self.to_cpu(gpu_airways_limited)
        # else:
        #     airways_limited = (airways_big & convex_body).astype(np.uint8)
        # 
        # print(f"     Airways limited before cleaning: shape={airways_limited.shape}, voxels={airways_limited.sum()}")
        # airways_final = clean_airways_configurable(airways_limited, lungs_final)
        # print(f"     Airways after cleaning: shape={airways_final.shape}, voxels={airways_final.sum()}")
        # masks['airways'] = airways_final
        # print(f"     Airways mask: shape={airways_final.shape}, voxels={airways_final.sum()}")
        
        # Создаем пустую маску airways для совместимости
        airways_final = np.zeros_like(volume, dtype=np.uint8)
        masks['airways'] = airways_final
        
        # 6. Обновляем body маску согласно новой логике
        print("     Обновление body маски...")
        if lungs_final.any():
            # Если есть легкие, body = small_body ИЛИ lungs_final
            body_final = np.logical_or(small_body, lungs_final).astype(np.uint8)
        else:
            # Если нет легких, body = big_body
            body_final = big_body.copy()
        
        masks['body'] = body_final
        print(f"     Body mask: shape={body_final.shape}, voxels={body_final.sum()}")
        
        # 7. Soft ткани = small_body
        print("     Создание soft маски...")
        soft_mask = small_body.copy()
        masks['soft'] = soft_mask
        print(f"     Soft mask: shape={soft_mask.shape}, voxels={soft_mask.sum()}")
        
        
        # 8. Деление костей (если требуется)
        if args.separate_bones and args.divide_bones and bone_final is not None:
            print("     Разделение костей...")
            spine_mask, ribs_mask = separate_bones_configurable(bone_final)
            masks['spine'] = spine_mask
            masks['ribs'] = ribs_mask
        
        print(f"✅ Оптимизированная GPU сегментация завершена")
        return masks


def create_optimized_gpu_masks(volume: np.ndarray, projector, args, case_name=None, use_gpu: bool = True) -> Dict[str, np.ndarray]:
    """
    Создаёт конфигурируемые маски с оптимизированным GPU ускорением
    
    Args:
        volume: 3D массив CT данных
        projector: Проектор для сегментации
        args: Аргументы конфигурации
        case_name: Имя кейса
        use_gpu: Использовать ли GPU ускорение
    
    Returns:
        Словарь с масками компонент
    """
    processor = OptimizedGPUSegmentationProcessor(use_gpu=use_gpu)
    return processor.create_optimized_gpu_masks(volume, projector, args, case_name)


def process_single_case_optimized_gpu(case_name, data_dir, output_dir, args, use_gpu: bool = True):
    """Оптимизированная GPU обработка одного кейса"""
    
    print(f"🔥 Оптимизированная GPU обработка кейса: {case_name}")
    print(f"   GPU: {'включен' if use_gpu else 'отключен'}")
    
    # Загрузка данных
    print(f"1. Загрузка данных для {case_name}...")
    visualizer = CTVisualizer(data_dir, output_dir)
    visualizer.load_data()
    volume = visualizer.volume
    
    print(f"   Диапазон HU: [{volume.min():.0f}, {volume.max():.0f}]")
    print(f"   Размер: {volume.shape} ({volume.nbytes / 1024**3:.2f} GB)")
    
    # GPU сегментация
    print("\n2. Оптимизированная GPU сегментация...")
    start_time = time.time()
    
    masks = create_optimized_gpu_masks(volume, visualizer.projector, args, case_name, use_gpu)
    
    segmentation_time = time.time() - start_time
    print(f"   Время сегментации: {segmentation_time:.2f} сек")
    
    # Анализ результатов
    print("\n3. Анализ результатов...")
    analyze_optimized_results(volume, masks, args, case_name, segmentation_time)
    
    # Создание визуализаций
    print("\n4. Создание визуализаций...")
    create_optimized_visualizations(volume, masks, visualizer.metadata, output_dir, args, case_name)
    
    print(f"\n🔥 ОПТИМИЗИРОВАННАЯ GPU ОБРАБОТКА {case_name} ЗАВЕРШЕНА!")
    return True


def analyze_optimized_results(volume: np.ndarray, masks: Dict[str, np.ndarray], args, case_name: str, segmentation_time: float):
    """Анализ результатов оптимизированной GPU сегментации"""
    
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


def create_optimized_visualizations(volume: np.ndarray, masks: Dict[str, np.ndarray], metadata: Dict, 
                                   output_dir: Path, args, case_name: str):
    """Создание визуализаций для оптимизированной GPU сегментации"""
    
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
        fig.suptitle(f'🔥 ОПТИМИЗИРОВАННАЯ GPU СЕГМЕНТАЦИЯ: {case_name}', 
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
        out = output_dir / f"{case_name}_optimized_gpu_segmentation.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   🔥 Оптимизированная GPU визуализация: {out.name}")
        
    except Exception as e:
        print(f"   ⚠️  Ошибка создания визуализации: {e}")


if __name__ == '__main__':
    # Тестирование оптимизированной GPU сегментации
    parser = argparse.ArgumentParser(description='Оптимизированная GPU сегментация CT')
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
        output_dir = Path(__file__).parent / 'optimized_gpu_visualizations'
    
    output_dir.mkdir(exist_ok=True)
    
    use_gpu = not args.no_gpu and GPU_AVAILABLE
    
    if args.case:
        case_dir = data_root / args.case
        if case_dir.exists():
            success = process_single_case_optimized_gpu(args.case, case_dir, output_dir, args, use_gpu)
            print(f"Результат: {'Успех' if success else 'Ошибка'}")
        else:
            print(f"Кейс не найден: {case_dir}")
    else:
        print("Укажите кейс с --case")

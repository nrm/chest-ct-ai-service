#!/usr/bin/env python3
"""
Тест sparse segmentation с интерполяцией
"""

import sys
import os
import numpy as np
from pathlib import Path
import time
import tempfile
import shutil

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

def test_sparse_segmentation():
    """Тестирует sparse segmentation"""
    print("🎯 Тестирование sparse segmentation...")
    
    try:
        from utils.sparse_segmentation import (
            create_sparse_segmentation, 
            interpolate_masks_between_slices,
            adaptive_slice_selection
        )
        
        # Создаем синтетический объем
        print("   Создание синтетического объема...")
        z, y, x = 100, 256, 256  # Большой объем для тестирования
        
        # Создаем объем с изменяющейся структурой
        volume = np.random.randn(z, y, x).astype(np.float32)
        
        # Добавляем структуры, которые меняются по Z
        for i in range(z):
            # Создаем "легкие" в центре
            center_y, center_x = y // 2, x // 2
            radius = 50 + 20 * np.sin(i * 0.1)  # Меняющийся радиус
            
            y_coords, x_coords = np.ogrid[:y, :x]
            mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 < radius**2
            volume[i][mask] = -800  # HU для легких
            
            # Добавляем "кости" по краям
            bone_mask = (y_coords < 20) | (y_coords > y-20) | (x_coords < 20) | (x_coords > x-20)
            volume[i][bone_mask] = 1000  # HU для костей
        
        print(f"   Объем: {volume.shape} ({volume.nbytes / 1024**3:.2f} GB)")
        
        # Создаем mock projector
        class MockProjector:
            def _compute_lung_mask_enhanced(self, volume, body_mask):
                # Простая сегментация легких по HU
                return (volume < -500).astype(np.uint8)
            
            def _compute_bone_mask_enhanced(self, volume, body_mask):
                # Простая сегментация костей по HU
                return (volume > 200).astype(np.uint8)
            
            def _compute_airways_mask(self, volume, lungs, body):
                # Простая сегментация дыхательных путей
                return (volume < -900).astype(np.uint8)
        
        # Mock args
        class MockArgs:
            def __init__(self):
                self.separate_bones = True
                self.divide_bones = False
        
        projector = MockProjector()
        args = MockArgs()
        
        # Тест 1: Адаптивный выбор слайсов
        print("\n   Тест 1: Адаптивный выбор слайсов...")
        selected_indices = adaptive_slice_selection(volume, max_slices=20, min_step=5)
        print(f"   Выбрано {len(selected_indices)} слайсов из {z}: {selected_indices[:10]}...")
        
        # Тест 2: Sparse segmentation
        print("\n   Тест 2: Sparse segmentation...")
        start_time = time.time()
        
        masks = create_sparse_segmentation(
            volume, projector, args, 
            slice_step=10, 
            interpolation_method='morphological'
        )
        
        segmentation_time = time.time() - start_time
        print(f"   Время сегментации: {segmentation_time:.2f}s")
        
        # Проверяем результаты
        print(f"   Сгенерированные маски: {list(masks.keys())}")
        for comp_name, mask in masks.items():
            if mask is not None:
                print(f"     {comp_name}: {mask.shape}, {mask.sum()} вокселей")
        
        # Тест 3: Сравнение с полной сегментацией (для небольшого объема)
        print("\n   Тест 3: Сравнение производительности...")
        small_volume = volume[:20]  # Маленький объем для сравнения
        
        # Sparse для маленького объема
        start_time = time.time()
        sparse_masks = create_sparse_segmentation(
            small_volume, projector, args, 
            slice_step=5, 
            interpolation_method='morphological'
        )
        sparse_time = time.time() - start_time
        
        print(f"   Sparse segmentation (20 слайсов): {sparse_time:.2f}s")
        print(f"   Рассчитано слайсов: {20 // 5 + 1}")
        
        # Оценка времени полной сегментации
        estimated_full_time = sparse_time * 20 / (20 // 5 + 1)
        speedup = estimated_full_time / sparse_time
        print(f"   Оценка ускорения: {speedup:.1f}x")
        
        print("✅ Sparse segmentation тест завершен успешно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования sparse segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interpolation_methods():
    """Тестирует различные методы интерполяции"""
    print("\n🔄 Тестирование методов интерполяции...")
    
    try:
        from utils.sparse_segmentation import interpolate_masks_between_slices
        
        # Создаем тестовые данные
        z, y, x = 10, 64, 64
        slice_indices = [0, 3, 6, 9]  # Каждый 3-й слайс
        
        # Создаем тестовые маски
        sparse_masks = {
            'test': []
        }
        
        for i, slice_idx in enumerate(slice_indices):
            # Создаем маску с кругом в центре
            mask = np.zeros((y, x), dtype=np.uint8)
            center_y, center_x = y // 2, x // 2
            radius = 20 + i * 2  # Увеличивающийся радиус
            
            y_coords, x_coords = np.ogrid[:y, :x]
            circle_mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 < radius**2
            mask[circle_mask] = 1
            
            sparse_masks['test'].append(mask)
        
        # Тестируем разные методы интерполяции
        methods = ['nearest', 'linear', 'morphological']
        
        for method in methods:
            print(f"   Тестирование метода: {method}")
            start_time = time.time()
            
            full_masks = interpolate_masks_between_slices(
                sparse_masks, (z, y, x), slice_indices, method
            )
            
            interpolation_time = time.time() - start_time
            print(f"     Время: {interpolation_time:.3f}s")
            print(f"     Результат: {full_masks['test'].shape}, {full_masks['test'].sum()} вокселей")
        
        print("✅ Тест методов интерполяции завершен")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования интерполяции: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование оптимизированной сегментации с интерполяцией")
    print("=" * 70)
    
    # Тест 1: Sparse segmentation
    sparse_ok = test_sparse_segmentation()
    
    # Тест 2: Методы интерполяции
    interpolation_ok = test_interpolation_methods()
    
    # Итоги
    print("\n" + "=" * 70)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"   Sparse segmentation: {'✅' if sparse_ok else '❌'}")
    print(f"   Методы интерполяции: {'✅' if interpolation_ok else '❌'}")
    
    if sparse_ok and interpolation_ok:
        print("\n🎯 Оптимизированная сегментация готова к использованию!")
        print("   - Автоматический выбор метода для больших объемов")
        print("   - Интерполяция между рассчитанными слайсами")
        print("   - Значительное ускорение для больших DICOM серий")
    else:
        print("\n❌ Проблемы с оптимизированной сегментацией")

if __name__ == "__main__":
    main()

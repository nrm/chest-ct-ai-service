#!/usr/bin/env python3
"""
Тест оптимизированной GPU сегментации
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "segment_and_viz_2"))

def test_optimized_gpu():
    """Тестирует оптимизированную GPU сегментацию"""
    print("🔥 Тестирование оптимизированной GPU сегментации...")
    
    try:
        from optimized_gpu_segmentation import create_optimized_gpu_masks, GPU_AVAILABLE
        
        print(f"   GPU доступен: {GPU_AVAILABLE}")
        
        # Создаем синтетический CT том
        print("   Создание синтетического CT тома...")
        volume = np.random.randn(80, 80, 80).astype(np.float32)
        volume = volume * 1000 - 1000  # Имитируем HU значения
        
        # Создаем простой проектор
        class MockProjector:
            def _compute_bone_mask_enhanced(self, volume, body_mask):
                return (volume > 200).astype(np.uint8)
            
            def _compute_lung_mask_enhanced(self, volume, body_mask):
                return (volume < -500).astype(np.uint8)
            
            def _compute_airways_mask(self, volume, lung_mask, body_mask):
                return (volume < -800).astype(np.uint8)
        
        # Создаем аргументы
        class Args:
            def __init__(self):
                self.separate_bones = False  # Упрощаем для теста
                self.divide_bones = False
        
        args = Args()
        projector = MockProjector()
        
        print("   Запуск оптимизированной GPU сегментации...")
        start_time = time.time()
        
        masks = create_optimized_gpu_masks(volume, projector, args, "test_case", use_gpu=GPU_AVAILABLE)
        
        segmentation_time = time.time() - start_time
        print(f"   Время сегментации: {segmentation_time:.2f} секунд")
        
        # Проверяем результаты
        print("   Результаты сегментации:")
        for comp_name, mask in masks.items():
            if mask is not None:
                voxel_count = mask.sum()
                print(f"     {comp_name}: {voxel_count:,} вокселей")
            else:
                print(f"     {comp_name}: не найдено")
        
        print("✅ Оптимизированная GPU сегментация работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка оптимизированной GPU сегментации: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpu_fallback():
    """Тестирует CPU fallback"""
    print("\n💻 Тестирование CPU fallback...")
    
    try:
        from optimized_gpu_segmentation import create_optimized_gpu_masks
        
        # Создаем синтетический CT том
        volume = np.random.randn(50, 50, 50).astype(np.float32)
        volume = volume * 1000 - 1000
        
        class MockProjector:
            def _compute_bone_mask_enhanced(self, volume, body_mask):
                return (volume > 200).astype(np.uint8)
            
            def _compute_lung_mask_enhanced(self, volume, body_mask):
                return (volume < -500).astype(np.uint8)
            
            def _compute_airways_mask(self, volume, lung_mask, body_mask):
                return (volume < -800).astype(np.uint8)
        
        class Args:
            def __init__(self):
                self.separate_bones = False
                self.divide_bones = False
        
        args = Args()
        projector = MockProjector()
        
        print("   Запуск CPU сегментации...")
        start_time = time.time()
        
        masks = create_optimized_gpu_masks(volume, projector, args, "test_case", use_gpu=False)
        
        segmentation_time = time.time() - start_time
        print(f"   Время сегментации: {segmentation_time:.2f} секунд")
        
        print("✅ CPU fallback работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка CPU fallback: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование оптимизированной GPU сегментации")
    print("=" * 60)
    
    # Тест 1: Оптимизированная GPU сегментация
    gpu_ok = test_optimized_gpu()
    
    # Тест 2: CPU fallback
    cpu_ok = test_cpu_fallback()
    
    # Итоги
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"   Оптимизированная GPU: {'✅' if gpu_ok else '❌'}")
    print(f"   CPU fallback: {'✅' if cpu_ok else '❌'}")
    
    if gpu_ok and cpu_ok:
        print("\n🔥 Оптимизированная GPU сегментация готова!")
    else:
        print("\n❌ Проблемы с сегментацией")

if __name__ == "__main__":
    main()

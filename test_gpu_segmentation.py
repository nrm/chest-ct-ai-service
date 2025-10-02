#!/usr/bin/env python3
"""
Тестовый скрипт для проверки GPU сегментации
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "segment_and_viz_2"))

def test_gpu_availability():
    """Тестирует доступность GPU"""
    print("🔥 Тестирование GPU доступности...")
    
    try:
        import cupy as cp
        print(f"✅ CuPy доступен: версия {cp.__version__}")
        print(f"   CUDA версия: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"   Количество устройств: {cp.cuda.runtime.getDeviceCount()}")
        
        if cp.cuda.runtime.getDeviceCount() > 0:
            memory_info = cp.cuda.runtime.memGetInfo()
            total_memory = memory_info[1] // (1024**3)
            free_memory = memory_info[0] // (1024**3)
            print(f"   Память GPU: {free_memory}GB свободно из {total_memory}GB")
            
            # Простой тест
            print("   Тестирование простых операций...")
            a = cp.array([1, 2, 3, 4, 5])
            b = cp.array([5, 4, 3, 2, 1])
            c = a + b
            print(f"   Результат теста: {c.get()}")
            print("✅ GPU работает корректно")
            return True
        else:
            print("❌ GPU устройства не найдены")
            return False
            
    except ImportError:
        print("❌ CuPy не установлен")
        return False
    except Exception as e:
        print(f"❌ Ошибка GPU: {e}")
        return False

def test_gpu_segmentation():
    """Тестирует GPU сегментацию на синтетических данных"""
    print("\n🧠 Тестирование GPU сегментации...")
    
    try:
        from gpu_segmentation import create_gpu_configurable_masks, GPU_AVAILABLE
        from configurable_dual_body_sementation import CTVisualizer
        
        if not GPU_AVAILABLE:
            print("❌ GPU недоступен для сегментации")
            return False
        
        # Создаем синтетический CT том
        print("   Создание синтетического CT тома...")
        volume = np.random.randn(100, 100, 100).astype(np.float32)
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
                self.separate_bones = True
                self.divide_bones = False
        
        args = Args()
        projector = MockProjector()
        
        print("   Запуск GPU сегментации...")
        start_time = time.time()
        
        masks = create_gpu_configurable_masks(volume, projector, args, "test_case", use_gpu=True)
        
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
        
        print("✅ GPU сегментация работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка GPU сегментации: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpu_fallback():
    """Тестирует fallback на CPU"""
    print("\n💻 Тестирование CPU fallback...")
    
    try:
        from gpu_segmentation import create_gpu_configurable_masks
        
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
        
        masks = create_gpu_configurable_masks(volume, projector, args, "test_case", use_gpu=False)
        
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
    print("🚀 Тестирование GPU сегментации RadiAssist")
    print("=" * 50)
    
    # Тест 1: Доступность GPU
    gpu_available = test_gpu_availability()
    
    # Тест 2: GPU сегментация
    if gpu_available:
        gpu_segmentation_ok = test_gpu_segmentation()
    else:
        print("\n⚠️  Пропуск GPU сегментации (GPU недоступен)")
        gpu_segmentation_ok = False
    
    # Тест 3: CPU fallback
    cpu_fallback_ok = test_cpu_fallback()
    
    # Итоги
    print("\n" + "=" * 50)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"   GPU доступен: {'✅' if gpu_available else '❌'}")
    print(f"   GPU сегментация: {'✅' if gpu_segmentation_ok else '❌'}")
    print(f"   CPU fallback: {'✅' if cpu_fallback_ok else '❌'}")
    
    if gpu_available and gpu_segmentation_ok:
        print("\n🔥 GPU сегментация готова к использованию!")
    elif cpu_fallback_ok:
        print("\n💻 CPU сегментация работает (GPU недоступен)")
    else:
        print("\n❌ Проблемы с сегментацией")

if __name__ == "__main__":
    main()

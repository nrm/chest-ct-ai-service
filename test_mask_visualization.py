#!/usr/bin/env python3
"""
Тест визуализации масок на DICOM слайсах
"""

import sys
import os
from pathlib import Path
import numpy as np
import tempfile
import shutil

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

def test_mask_visualization():
    """Тестирует визуализацию масок"""
    print("🎨 Тестирование визуализации масок...")
    
    try:
        from utils.mask_visualization import (
            create_annotated_slice, create_multi_mask_overlay, 
            apply_window_level, create_comparison_view
        )
        
        # Создаем синтетический DICOM слайс
        print("   Создание синтетического DICOM слайса...")
        slice_data = np.random.randn(256, 256).astype(np.float32)
        slice_data = slice_data * 1000 - 1000  # Имитируем HU значения
        
        # Создаем синтетические маски
        masks = {
            'lungs': np.zeros((256, 256), dtype=np.uint8),
            'bone': np.zeros((256, 256), dtype=np.uint8),
            'soft': np.zeros((256, 256), dtype=np.uint8)
        }
        
        # Добавляем "легкие" (низкие HU значения)
        masks['lungs'][100:150, 80:180] = 1
        
        # Добавляем "кости" (высокие HU значения)
        masks['bone'][50:200, 50:60] = 1
        masks['bone'][50:200, 190:200] = 1
        
        # Добавляем "мягкие ткани"
        masks['soft'][120:140, 100:160] = 1
        
        # Создаем временный DICOM файл
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp_file:
            tmp_dicom_path = Path(tmp_file.name)
        
        # Создаем простой DICOM файл
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        
        # Создаем базовый DICOM dataset
        ds = Dataset()
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        ds.SOPInstanceUID = '1.2.3.4.5.6.7.8.9'
        ds.StudyInstanceUID = '1.2.3.4.5.6.7.8.9.1'
        ds.SeriesInstanceUID = '1.2.3.4.5.6.7.8.9.2'
        ds.Modality = 'CT'
        ds.PatientName = 'Test^Patient'
        ds.PatientID = 'TEST001'
        ds.StudyDate = '20240101'
        ds.StudyTime = '120000'
        ds.SeriesDate = '20240101'
        ds.SeriesTime = '120000'
        ds.InstanceNumber = 1
        ds.ImagePositionPatient = [0, 0, 0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceThickness = 1.0
        ds.PixelSpacing = [1.0, 1.0]
        ds.Rows = 256
        ds.Columns = 256
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.WindowCenter = 0
        ds.WindowWidth = 400
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        
        # Добавляем пиксельные данные
        ds.PixelData = slice_data.astype(np.int16).tobytes()
        
        # Сохраняем DICOM файл
        ds.save_as(str(tmp_dicom_path))
        
        # Создаем выходную директорию
        output_dir = Path("test_mask_output")
        output_dir.mkdir(exist_ok=True)
        
        # Тест 1: Аннотированный слайс
        print("   Тест 1: Аннотированный слайс...")
        annotated_path = output_dir / "annotated_slice.png"
        success = create_annotated_slice(tmp_dicom_path, masks, 0, annotated_path)
        
        if success:
            print(f"     ✅ Аннотированный слайс: {annotated_path}")
        else:
            print("     ❌ Ошибка создания аннотированного слайса")
        
        # Тест 2: Сравнительный вид
        print("   Тест 2: Сравнительный вид...")
        comparison_path = output_dir / "comparison_view.png"
        success = create_comparison_view(tmp_dicom_path, masks, 0, comparison_path)
        
        if success:
            print(f"     ✅ Сравнительный вид: {comparison_path}")
        else:
            print("     ❌ Ошибка создания сравнительного вида")
        
        # Тест 3: Наложение масок
        print("   Тест 3: Наложение масок...")
        processed_image = apply_window_level(slice_data, 0, 400)
        overlay_image = create_multi_mask_overlay(processed_image, masks)
        
        # Сохраняем наложение
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay_image)
        plt.title('Mask Overlay Test')
        plt.axis('off')
        overlay_path = output_dir / "mask_overlay.png"
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Наложение масок: {overlay_path}")
        
        # Очистка
        tmp_dicom_path.unlink()
        
        print("✅ Тестирование визуализации масок завершено")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования визуализации: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dicom_slice_generation():
    """Тестирует генерацию слайсов для задачи"""
    print("\n📁 Тестирование генерации слайсов для задачи...")
    
    try:
        from utils.mask_visualization import generate_mask_slices_for_task
        
        # Создаем временную директорию с DICOM файлами
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Создаем несколько синтетических DICOM файлов
            dicom_files = []
            for i in range(5):
                dicom_file = temp_path / f"slice_{i:04d}.dcm"
                
                # Создаем синтетический слайс
                slice_data = np.random.randn(128, 128).astype(np.float32)
                slice_data = slice_data * 1000 - 1000
                
                # Создаем простой DICOM файл
                import pydicom
                ds = pydicom.Dataset()
                ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                ds.SOPInstanceUID = f'1.2.3.4.5.6.7.8.9.{i}'
                ds.Modality = 'CT'
                ds.Rows = 128
                ds.Columns = 128
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                ds.PhotometricInterpretation = 'MONOCHROME2'
                ds.WindowCenter = 0
                ds.WindowWidth = 400
                ds.PixelData = slice_data.astype(np.int16).tobytes()
                
                ds.save_as(str(dicom_file))
                dicom_files.append(dicom_file)
            
            # Создаем синтетические маски
            masks = {
                'lungs': np.random.randint(0, 2, (5, 128, 128), dtype=np.uint8),
                'bone': np.random.randint(0, 2, (5, 128, 128), dtype=np.uint8)
            }
            
            # Генерируем слайсы
            output_dir = Path("test_slices_output")
            generated_slices = generate_mask_slices_for_task(
                "test_task", dicom_files, masks, output_dir, max_slices=3
            )
            
            print(f"   Сгенерировано слайсов: {len(generated_slices)}")
            for slice_info in generated_slices:
                print(f"     - {slice_info['filename']} (slice {slice_info['slice_index']})")
            
            print("✅ Тестирование генерации слайсов завершено")
            return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования генерации слайсов: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование визуализации масок на DICOM слайсах")
    print("=" * 60)
    
    # Тест 1: Визуализация масок
    viz_ok = test_mask_visualization()
    
    # Тест 2: Генерация слайсов
    slices_ok = test_dicom_slice_generation()
    
    # Итоги
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"   Визуализация масок: {'✅' if viz_ok else '❌'}")
    print(f"   Генерация слайсов: {'✅' if slices_ok else '❌'}")
    
    if viz_ok and slices_ok:
        print("\n🎨 Визуализация масок готова к использованию!")
    else:
        print("\n❌ Проблемы с визуализацией")

if __name__ == "__main__":
    main()

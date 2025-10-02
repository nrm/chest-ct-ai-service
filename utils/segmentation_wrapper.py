"""
Wrapper для интеграции сегментации из segment_and_viz_2
Использует configurable_dual_body_sementation.py для быстрой сегментации body и lungs
"""
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import tempfile
import shutil
import argparse
import os

# Добавляем путь к модулям сегментации
SEGMENT_DIR = Path(__file__).parent.parent / "segment_and_viz_2"
sys.path.insert(0, str(SEGMENT_DIR))
print(f"📁 Added segmentation path: {SEGMENT_DIR}")
print(f"   Exists: {SEGMENT_DIR.exists()}")

try:
    from configurable_dual_body_sementation import CTVisualizer
    from optimized_gpu_segmentation import create_optimized_gpu_masks, GPU_AVAILABLE
    SEGMENTATION_AVAILABLE = True
    print(f"🔥 Оптимизированная GPU сегментация: {'доступна' if GPU_AVAILABLE else 'недоступна'}")
except ImportError as e:
    print(f"⚠️  Segmentation not available: {e}")
    SEGMENTATION_AVAILABLE = False
    GPU_AVAILABLE = False

# Импорт ct_lung интеграции
try:
    from ct_lung_integration import create_enhanced_bones_mask, get_ct_lung_status
    CT_LUNG_INTEGRATION_AVAILABLE = True
    ct_lung_status = get_ct_lung_status()
    print(f"🦴 ct_lung.py интеграция: {'доступна' if ct_lung_status['available'] else 'недоступна'}")
except ImportError as e:
    print(f"⚠️  ct_lung integration not available: {e}")
    CT_LUNG_INTEGRATION_AVAILABLE = False


class SegmentationProcessor:
    """Обработчик сегментации для DICOM данных"""
    
    def __init__(self, task_id: str, dicom_dir: Path, output_dir: Path, include_bones: bool = False):
        self.task_id = task_id
        self.dicom_dir = dicom_dir
        self.output_dir = output_dir
        self.include_bones = include_bones
        self.masks_dir = output_dir / "masks"
        self.masks_dir.mkdir(exist_ok=True, parents=True)
        self.masks_metadata = {} # Initialize masks_metadata here
        print(f"     SegmentationProcessor initialized with dicom_dir: {self.dicom_dir}")
        
    def process(self) -> Optional[Dict]:
        """Запускает GPU-оптимизированную сегментацию и возвращает метаданные масок"""
        if not SEGMENTATION_AVAILABLE:
            print(f"⚠️  Segmentation skipped for task {self.task_id}: dependencies not available")
            return None
            
        try:
            print(f"🧠 Starting GPU segmentation for task {self.task_id}...")
            print(f"   Include bones: {self.include_bones}")
            print(f"   GPU available: {GPU_AVAILABLE}")
            
            # Создаём визуализатор для загрузки данных
            visualizer = CTVisualizer(self.dicom_dir, self.output_dir)
            visualizer.load_data()
            volume = visualizer.volume
            
            if volume is None:
                print(f"❌ Failed to load volume for task {self.task_id}")
                return None
            
            print(f"✅ Loaded volume: shape={volume.shape} ({volume.nbytes / 1024**3:.2f} GB)")
            
            # Создаём аргументы для сегментации
            class Args:
                def __init__(self, separate_bones=False, divide_bones=False):
                    self.separate_bones = separate_bones
                    self.divide_bones = divide_bones
            
            args = Args(separate_bones=self.include_bones, divide_bones=False)
            
            # Используем полную сегментацию для всех объемов
            z, y, x = volume.shape
            total_slices = z
            
            print(f"🔥 Полная GPU сегментация для {total_slices} слайсов...")
            import time
            start_time = time.time()
            masks = create_optimized_gpu_masks(volume, visualizer.projector, args, self.task_id, use_gpu=GPU_AVAILABLE)
            
            segmentation_time = time.time() - start_time
            print(f"⚡ GPU segmentation time: {segmentation_time:.2f} seconds")
            
            # ВСЕГДА улучшаем сегментацию костей с помощью ct_lung.py после основной сегментации
            if CT_LUNG_INTEGRATION_AVAILABLE:
                print("🦴 Улучшение сегментации костей с ct_lung.py (всегда после основной сегментации)...")
                try:
                    # Получаем spacing из метаданных
                    spacing_zyx = visualizer.metadata.get('spacing', (1.0, 1.0, 1.0))
                    if len(spacing_zyx) == 3:
                        spacing_zyx = tuple(spacing_zyx)
                    else:
                        spacing_zyx = (1.0, 1.0, 1.0)
                    
                    # Создаем улучшенную маску костей (всегда, независимо от include_bones)
                    enhanced_bones = create_enhanced_bones_mask(
                        volume, spacing_zyx, masks.get('body', np.ones_like(volume, dtype=bool)), 
                        masks.get('bone')  # Может быть None, если include_bones=False
                    )
                    
                    # Обновляем маску костей
                    masks['bone'] = enhanced_bones
                    print(f"✅ Улучшенная маска костей: {enhanced_bones.sum()} вокселей")
                    
                except Exception as e:
                    print(f"⚠️ Ошибка улучшения костей: {e}")
            else:
                print("⚠️ ct_lung.py недоступен, используем только основную сегментацию")
            
            # Сохраняем маски
            masks_metadata = self._save_masks(masks, volume, visualizer.metadata)
            
            # Добавляем информацию о GPU и методе сегментации в метаданные
            masks_metadata["gpu_info"] = {
                "gpu_used": GPU_AVAILABLE,
                "segmentation_time": segmentation_time,
                "volume_size_gb": volume.nbytes / 1024**3,
                "segmentation_method": "full",
                "total_slices": total_slices,
                "slice_step": 1,
                "calculated_slices": total_slices,
                "ct_lung_enhanced": CT_LUNG_INTEGRATION_AVAILABLE
            }
            
            # Сохраняем метаданные для использования в других функциях
            self.masks_metadata = masks_metadata
            
            # Создаём preview с наложением масок
            self._create_overlay_preview(volume, masks, visualizer.metadata)
            
            # Генерируем визуализированные слайсы DICOM
            self._generate_mask_slices(volume, masks, visualizer.metadata)
            
            print(f"✅ GPU segmentation completed for task {self.task_id}")
            return masks_metadata
            
        except Exception as e:
            print(f"❌ GPU segmentation failed for task {self.task_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_masks(self, masks: Dict[str, np.ndarray], volume: np.ndarray, 
                    metadata: Dict) -> Dict:
        """Сохраняет маски в форматах для 2D и 3D визуализации"""
        print(f"     Saving masks: {list(masks.keys())}")
        masks_metadata = {
            "task_id": self.task_id,
            "volume_shape": list(volume.shape),
            "spacing": [float(x) for x in metadata.get('spacing', [1.0, 1.0, 1.0])],
            "components": {}
        }
        
        # Список компонентов для сохранения (только те, что есть в masks)
        components = list(masks.keys())
        print(f"     Components to process: {components}")
        
        for comp_name in components:
            mask = masks[comp_name]
            if mask is None:
                print(f"     Skipping {comp_name}: mask is None")
                continue
            
            voxel_count = mask.sum() if hasattr(mask, 'sum') else 0
            print(f"     Processing {comp_name}: shape={mask.shape}, voxels={voxel_count}")
            
            if voxel_count == 0:
                print(f"     Warning: {comp_name} mask is empty, but saving metadata anyway")
                # Сохраняем метаданные даже для пустых масок
                masks_metadata["components"][comp_name] = {
                    "mask_3d_file": None,
                    "slices_dir": None,
                    "slice_indices": [],
                    "voxel_count": 0,
                    "volume_ml": 0.0
                }
                continue
            
            # Сохраняем как numpy array для 3D визуализации
            mask_3d_path = self.masks_dir / f"{comp_name}_3d.npy"
            np.save(str(mask_3d_path), mask.astype(np.uint8))
            
            # Сохраняем срезы для 2D overlay
            slices_dir = self.masks_dir / comp_name
            slices_dir.mkdir(exist_ok=True)
            
            # Сохраняем каждый срез (убрали оптимизацию "каждый 5-й")
            slice_indices = []
            for z in range(mask.shape[0]):
                if mask[z].any():
                    slice_path = slices_dir / f"slice_{z:04d}.png"
                    self._save_mask_slice_as_png(mask[z], slice_path)
                    slice_indices.append(z)
            
            # Метаданные компонента
            masks_metadata["components"][comp_name] = {
                "mask_3d_file": str(mask_3d_path.relative_to(self.output_dir)),
                "slices_dir": str(slices_dir.relative_to(self.output_dir)),
                "slice_indices": slice_indices,
                "voxel_count": int(mask.sum()),
                "volume_ml": float(mask.sum() * np.prod(metadata.get('spacing', [1.0, 1.0, 1.0])) / 1000.0)
            }
        
        # Сохраняем метаданные
        metadata_path = self.masks_dir / "masks_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(masks_metadata, f, indent=2)
        
        return masks_metadata
    
    def _save_mask_slice_as_png(self, mask_slice: np.ndarray, output_path: Path):
        """Сохраняет срез маски как PNG"""
        from PIL import Image
        
        # Конвертируем в uint8 (255 для маски, 0 для фона)
        mask_img = (mask_slice.astype(np.uint8) * 255)
        
        # Сохраняем как PNG с прозрачностью
        img = Image.fromarray(mask_img, mode='L')
        img.save(str(output_path))
    
    def _create_overlay_preview(self, volume: np.ndarray, masks: Dict[str, np.ndarray], 
                                metadata: Dict):
        """Создаёт preview с наложением масок на срезы"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap, BoundaryNorm
            
            # Берём центральный аксиальный срез
            mid_z = volume.shape[0] // 2
            img_slice = volume[mid_z]
            
            # Создаём композитную маску с разными метками
            composite_mask = np.zeros_like(img_slice, dtype=np.uint8)
            
            # Присваиваем разные ID разным компонентам
            component_ids = {
                "body": 1,
                "lungs": 2,
                "bone": 3,
                "spine": 4,
                "ribs": 5,
                "soft": 6
            }
            
            # Сначала накладываем все компоненты кроме body
            for comp_name, comp_id in component_ids.items():
                if comp_name != "body" and comp_name in masks and masks[comp_name] is not None:
                    mask = masks[comp_name][mid_z]
                    composite_mask[mask > 0] = comp_id
            
            # Затем накладываем body только там, где нет других компонентов
            if "body" in masks and masks["body"] is not None:
                body_mask = masks["body"][mid_z]
                # body показывается только там, где нет других компонентов
                body_only = body_mask & (composite_mask == 0)
                composite_mask[body_only > 0] = component_ids["body"]
            
            # Создаём цветовую карту
            colors = [
                (0, 0, 0, 0),          # 0: background (transparent)
                (0.7, 0.7, 0.7, 0.3),  # 1: body (gray)
                (0.1, 0.7, 1.0, 0.4),  # 2: lungs (cyan)
                (1.0, 1.0, 0.2, 0.5),  # 3: bone (yellow)
                (1.0, 0.8, 0.4, 0.5),  # 4: spine (light yellow)
                (1.0, 1.0, 0.6, 0.4),  # 5: ribs (lighter yellow)
                (0.7, 0.3, 1.0, 0.3),  # 6: soft tissue (purple)
            ]
            cmap = ListedColormap(colors)
            
            # Визуализация
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Отображаем CT срез
            ax.imshow(np.clip(img_slice, -1000, 500), cmap='gray', 
                     vmin=-1000, vmax=500, aspect='auto')
            
            # Наложение масок
            ax.imshow(composite_mask, cmap=cmap, interpolation='nearest', 
                     vmin=0, vmax=len(colors)-1, aspect='auto')
            
            ax.set_title(f'Segmentation Preview - Slice {mid_z}/{volume.shape[0]}')
            ax.axis('off')
            
            # Добавляем легенду
            legend_elements = []
            for comp_name, comp_id in component_ids.items():
                if comp_name in masks and masks[comp_name] is not None:
                    color = colors[comp_id]
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color[:3], markersize=10, 
                                 label=comp_name.replace('_', ' ').title())
                    )
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right')
            
            # Сохраняем
            preview_path = self.output_dir / "segmentation_preview.png"
            fig.tight_layout()
            fig.savefig(str(preview_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Saved segmentation preview: {preview_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to create overlay preview: {e}")
    
    def _generate_mask_slices(self, volume: np.ndarray, masks: Dict[str, np.ndarray], metadata: Dict):
        """Генерирует PNG слайсы DICOM с наложением масок"""
        try:
            from utils.mask_visualization import generate_mask_slices_for_task
            from utils.dicom_to_image import get_dicom_files_sorted
            import SimpleITK as sitk
            
            # Получаем список DICOM файлов в том же порядке, что и при загрузке volume
            # Используем SimpleITK для правильной сортировки (как в CTVolumeLoader)
            try:
                reader = sitk.ImageSeriesReader()
                series_ids = list(reader.GetGDCMSeriesIDs(str(self.dicom_dir)))
                if series_ids:
                    # Берем первую серию (как в CTVolumeLoader)
                    dicom_files = [Path(f) for f in reader.GetGDCMSeriesFileNames(str(self.dicom_dir), series_ids[0])]
                    print(f"✅ Использован порядок SimpleITK: {len(dicom_files)} файлов")
                else:
                    dicom_files = get_dicom_files_sorted(self.dicom_dir)
                    print(f"⚠️ SimpleITK серии не найдены, использован InstanceNumber")
            except Exception as e:
                print(f"⚠️ Ошибка SimpleITK сортировки: {e}, использован InstanceNumber")
                dicom_files = get_dicom_files_sorted(self.dicom_dir)
            
            if not dicom_files:
                print(f"⚠️  No DICOM files found in {self.dicom_dir}")
                return
            
            # Создаем директорию для слайсов
            slices_dir = self.output_dir / "masks" / "mask_slices"
            slices_dir.mkdir(exist_ok=True, parents=True)
            
            # Отладочная информация о масках
            print(f"🔍 Отладка масок для генерации слайсов:")
            for name, mask in masks.items():
                if hasattr(mask, 'shape'):
                    print(f"  {name}: shape={mask.shape}, voxels={mask.sum() if hasattr(mask, 'sum') else 'N/A'}")
                else:
                    print(f"  {name}: type={type(mask)}")
            
            # Генерируем слайсы с масками (каждый 3-й)
            generated_slices = generate_mask_slices_for_task(
                self.task_id, dicom_files, masks, slices_dir, slice_step=3
            )
            
            # Сохраняем информацию о сгенерированных слайсах
            slices_info = {
                "total_dicom_files": len(dicom_files),
                "generated_slices": len(generated_slices),
                "slices_dir": str(slices_dir.relative_to(self.output_dir)),
                "slices": generated_slices
            }
            
            # Добавляем информацию о методе сегментации
            slices_info["segmentation_method"] = "full"
            slices_info["slice_step"] = 3
            slices_info["calculated_slices"] = len(generated_slices)
            
            # Сохраняем информацию о слайсах в файл
            slices_info_path = slices_dir / "slices_info.json"
            with open(slices_info_path, 'w') as f:
                json.dump(slices_info, f, indent=2)
            
            # Добавляем информацию о слайсах в метаданные сегментации
            if hasattr(self, 'masks_metadata'):
                self.masks_metadata["mask_slices"] = slices_info
                print(f"📊 Added mask_slices info to metadata: {len(generated_slices)} slices")
            
            print(f"✅ Generated {len(generated_slices)} mask slices")
            
        except Exception as e:
            print(f"⚠️  Failed to generate mask slices: {e}")


def run_segmentation(task_id: str, dicom_dir: Path, output_dir: Path, include_bones: bool = False) -> Optional[Dict]:
    """
    Удобная функция для запуска сегментации
    
    Args:
        task_id: ID задачи
        dicom_dir: Путь к директории с DICOM файлами
        output_dir: Путь для сохранения результатов
        include_bones: Включать ли сегментацию костей (медленно)
    
    Returns:
        Метаданные масок или None при ошибке
    """
    processor = SegmentationProcessor(task_id, dicom_dir, output_dir, include_bones)
    return processor.process()
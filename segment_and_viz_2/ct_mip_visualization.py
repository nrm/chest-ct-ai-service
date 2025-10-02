#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для создания MIP (Maximum Intensity Projection) проекций CT данных
с правильной анатомической ориентацией
"""

import os
import numpy as np
import pydicom
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
try:
    import SimpleITK as sitk
    _HAS_SITK = True
except Exception:
    _HAS_SITK = False
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from skimage.morphology import convex_hull_image, binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import warnings
warnings.filterwarnings('ignore')


class SegmentationHelper:
    """Вспомогательный класс для сегментации с адаптивными алгоритмами"""
    
    @staticmethod
    def adaptive_threshold_multiotsu(values: np.ndarray, classes: int = 3, 
                                   fallback_percentiles: Optional[List[float]] = None) -> np.ndarray:
        """
        Адаптивный multi-Otsu с фоллбэком на перцентили
        
        Args:
            values: массив значений для анализа
            classes: количество классов для разделения
            fallback_percentiles: перцентили для фоллбэка (по умолчанию равномерное разделение)
        
        Returns:
            массив порогов
        """
        if values.size < 100:
            # Слишком мало данных для Otsu
            if fallback_percentiles is None:
                fallback_percentiles = [100 * i / classes for i in range(1, classes)]
            return np.percentile(values, fallback_percentiles)
        
        try:
            # Пытаемся применить multi-Otsu
            thresholds = threshold_multiotsu(values, classes=classes)
            
            # Проверяем разумность порогов (должны быть монотонно возрастающими)
            if len(thresholds) > 1 and np.any(np.diff(thresholds) <= 0):
                raise ValueError("Non-monotonic thresholds")
                
            return thresholds
            
        except Exception:
            # Фоллбэк на перцентили
            if fallback_percentiles is None:
                fallback_percentiles = [100 * i / classes for i in range(1, classes)]
            
            return np.percentile(values, fallback_percentiles)
    
    @staticmethod
    def find_valley_threshold(values: np.ndarray, min_val: float, max_val: float, 
                            bins: int = 256) -> float:
        """
        Находит долину в гистограмме между min_val и max_val
        
        Args:
            values: массив значений
            min_val: минимальное значение диапазона
            max_val: максимальное значение диапазона
            bins: количество бинов гистограммы
        
        Returns:
            значение порога в долине
        """
        # Фильтруем значения в диапазоне
        mask = (values >= min_val) & (values <= max_val)
        if not np.any(mask):
            return (min_val + max_val) / 2
        
        filtered_values = values[mask]
        
        # Строим гистограмму
        hist, bin_edges = np.histogram(filtered_values, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Сглаживаем гистограмму
        from scipy.ndimage import gaussian_filter1d
        smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=1.0)
        
        # Находим минимум (долину)
        valley_idx = np.argmin(smoothed_hist)
        
        return bin_centers[valley_idx]
    
    @staticmethod
    def get_largest_components(mask: np.ndarray, n_components: int = 1) -> np.ndarray:
        """
        Оставляет только n крупнейших связных компонентов
        
        Args:
            mask: бинарная маска
            n_components: количество компонентов для сохранения
        
        Returns:
            маска с n крупнейшими компонентами
        """
        labeled, num_labels = ndimage.label(mask)
        
        if num_labels == 0:
            return mask
        
        # Вычисляем размеры компонентов
        component_sizes = ndimage.sum(mask, labeled, index=range(1, num_labels + 1))
        
        # Находим индексы крупнейших компонентов
        largest_indices = np.argsort(component_sizes)[::-1][:n_components]
        
        # Создаем новую маску
        result = np.zeros_like(mask, dtype=bool)
        for idx in largest_indices:
            result |= (labeled == (idx + 1))
        
        return result.astype(np.uint8)
    
    @staticmethod
    def validate_volume_ratios(masks: Dict[str, np.ndarray], 
                             expected_ratios: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
        """
        Проверяет, что доли объёмов масок находятся в ожидаемых диапазонах
        
        Args:
            masks: словарь масок {название: маска}
            expected_ratios: ожидаемые диапазоны {название: (мин_доля, макс_доля)}
        
        Returns:
            словарь результатов валидации {название: валидна}
        """
        total_volume = sum(mask.sum() for mask in masks.values())
        if total_volume == 0:
            return {name: False for name in masks.keys()}
        
        results = {}
        for name, mask in masks.items():
            volume_ratio = mask.sum() / total_volume
            if name in expected_ratios:
                min_ratio, max_ratio = expected_ratios[name]
                results[name] = min_ratio <= volume_ratio <= max_ratio
            else:
                results[name] = True  # Нет ограничений
        
        return results


class CTVolumeLoader:
    """Класс для загрузки CT томов из DICOM файлов"""
    
    def load_volume_sitk(self):
        """Пробует загрузить серию через SimpleITK с правильной ориентацией.
        Логирует все найденные серии, выбирает лучшую по критериям:
        1) Больше всего файлов (срезов)
        2) При прочих равных — минимальная толщина среза (spacing_z)
        """
        if not _HAS_SITK:
            return None, None
        try:
            reader = sitk.ImageSeriesReader()
            series_ids = list(reader.GetGDCMSeriesIDs(str(self.case_dir)))
            if not series_ids:
                return None, None

            # Соберём информацию по сериям
            series_info = []  # (sid, files, spacing_z, size_z)
            best_files = []
            for sid in series_ids:
                files = list(reader.GetGDCMSeriesFileNames(str(self.case_dir), sid))
                spacing_z = None
                size_z = len(files)
                # попытаемся быстро прочитать метаданные через SimpleITK
                try:
                    img = sitk.ReadImage(files)  # вся серия
                    sx, sy, sz = img.GetSpacing()  # (x,y,z)
                    spacing_z = float(sz)
                except Exception:
                    spacing_z = None
                series_info.append((sid, files, spacing_z, size_z))

            # Лог: какие серии нашли
            print("Найдены серии (SeriesInstanceUID, кол-во файлов, spacing_z):")
            for sid, files, spacing_z, size_z in series_info:
                print(f"  - {sid}: {size_z} файлов, spacing_z={spacing_z}")

            # Выбор лучшей серии
            def sort_key(item):
                sid, files, spacing_z, size_z = item
                return (-size_z, spacing_z if spacing_z is not None else 1e9)

            series_info.sort(key=sort_key)
            sid, best_files, spacing_z, size_z = series_info[0]
            print(f"→ Выбрана серия: {sid} (files={size_z}, spacing_z={spacing_z})")

            # Чтение выбранной серии
            reader.SetFileNames(best_files)
            image = reader.Execute()
            vol = sitk.GetArrayFromImage(image).astype('float32')  # (z,y,x)
            sx, sy, sz = image.GetSpacing()
            spacing_zyx = (sz, sy, sx)
            origin = image.GetOrigin()
            direction = image.GetDirection()
            meta = {
                'shape': vol.shape,
                'spacing': list(spacing_zyx),
                'orientation': list(direction),
                'origin': list(origin),
            }
            return vol, meta
        except Exception as e:
            print("SimpleITK загрузка не удалась:", e)
            return None, None


    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self.volume = None
        self.spacing = None
        self.origin = None
        self.orientation = None
        self.dicom_files = []
        
    def scan_dicom_files(self) -> List[Path]:
        """Рекурсивно сканирует директорию и находит DICOM файлы"""
        dicom_files = []
        for root, _, files in os.walk(self.case_dir):
            for filename in files:
                if filename.lower().endswith('.json'):
                    continue
                p = Path(root) / filename
                try:
                    pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                    dicom_files.append(p)
                except Exception:
                    continue
        return dicom_files
    
    def check_multiframe(self, dcm_path: Path) -> Tuple[bool, Optional[np.ndarray]]:
        """Проверяет и загружает многокадровый DICOM"""
        try:
            dcm = pydicom.dcmread(str(dcm_path))
            
            if hasattr(dcm, 'NumberOfFrames') and int(dcm.NumberOfFrames) > 1:
                num_frames = int(dcm.NumberOfFrames)
                pixel_array = dcm.pixel_array
                
                # Применяем rescale
                if hasattr(dcm, 'RescaleIntercept') and hasattr(dcm, 'RescaleSlope'):
                    intercept = float(dcm.RescaleIntercept)
                    slope = float(dcm.RescaleSlope)
                    pixel_array = pixel_array.astype(float) * slope + intercept
                
                return True, pixel_array
            
            return False, None
        except:
            return False, None
    
    def sort_slices(self, dicom_files: List[Path]) -> List[Path]:
        """Улучшенная сортировка слайсов с обработкой ошибок"""
        if not dicom_files:
            return []
            
        slice_info = []
        failed_files = 0
        
        for i, dcm_path in enumerate(dicom_files):
            try:
                dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                
                # Пробуем разные способы получения позиции
                position = None
                
                # Способ 1: ImagePositionPatient
                if hasattr(dcm, 'ImagePositionPatient') and dcm.ImagePositionPatient:
                    try:
                        position = float(dcm.ImagePositionPatient[2])
                    except (ValueError, IndexError, TypeError):
                        pass
                
                # Способ 2: SliceLocation
                if position is None and hasattr(dcm, 'SliceLocation') and dcm.SliceLocation:
                    try:
                        position = float(dcm.SliceLocation)
                    except (ValueError, TypeError):
                        pass
                
                # Способ 3: InstanceNumber
                if position is None and hasattr(dcm, 'InstanceNumber') and dcm.InstanceNumber:
                    try:
                        position = float(dcm.InstanceNumber)
                    except (ValueError, TypeError):
                        pass
                
                # Способ 4: порядковый номер файла
                if position is None:
                    position = i
                
                slice_info.append((position, dcm_path))
                
            except Exception:
                failed_files += 1
                continue
        
        if not slice_info:
            raise ValueError(f"Не удалось обработать ни одного из {len(dicom_files)} DICOM файлов")
        
        if failed_files > 0:
            print(f"   ⚠️ Не удалось обработать {failed_files} файлов из {len(dicom_files)}")
        
        # Сортируем по позиции
        slice_info.sort(key=lambda x: x[0])
        sorted_files = [path for _, path in slice_info]
        
        return sorted_files
    
    def load_volume(self) -> Tuple[np.ndarray, Dict]:
        """Загружает CT том из DICOM файлов"""
        self.dicom_files = self.scan_dicom_files()
        # Попробуем сначала SimpleITK (правильная ориентация/сборка серии)
        vol_meta = self.load_volume_sitk()
        if vol_meta and vol_meta[0] is not None:
            self.volume, meta = vol_meta
            self.spacing = np.array(meta['spacing'])
            # orientation может быть длиной 6 или 9; просто сохраняем как есть
            self.orientation = np.array(meta['orientation']) if meta.get('orientation') is not None else None
            self.origin = np.array(meta['origin']) if meta.get('origin') is not None else None
            print(f"✓ Загружено через SimpleITK: {self.volume.shape}")
            metadata = {
                'shape': self.volume.shape,
                'spacing': self.spacing,
                'orientation': self.orientation,
                'origin': self.origin
            }
            return self.volume, metadata

        
        if not self.dicom_files:
            raise ValueError("No DICOM files found")
        
        # Проверяем многокадровый формат
        is_multiframe, volume = self.check_multiframe(self.dicom_files[0])
        
        if is_multiframe:
            print(f"✓ Загружен многокадровый DICOM: {volume.shape}")
            self.volume = volume
            
            # Получаем метаданные из первого файла
            dcm = pydicom.dcmread(str(self.dicom_files[0]), stop_before_pixels=True)
            
        else:
            # Загружаем обычные DICOM файлы
            print(f"✓ Найдено {len(self.dicom_files)} DICOM файлов")
            
            # Сортируем слайсы
            sorted_files = self.sort_slices(self.dicom_files)
            
            if not sorted_files:
                raise ValueError("Не удалось отсортировать DICOM файлы")
            
            # Загружаем первый слайс для получения размеров
            try:
                dcm = pydicom.dcmread(str(sorted_files[0]))
            except Exception as e:
                raise ValueError(f"Не удалось загрузить первый DICOM файл: {e}")
            
            if not hasattr(dcm, 'pixel_array'):
                raise ValueError("Первый DICOM файл не содержит изображения")
            
            rows = dcm.Rows
            cols = dcm.Columns
            num_slices = len(sorted_files)
            
            print(f"   📐 Размеры: {num_slices} x {rows} x {cols}")
            
            # Создаем том
            volume = np.zeros((num_slices, rows, cols), dtype=np.float32)
            
            # Загружаем все слайсы
            failed_slices = 0
            for i, dcm_path in enumerate(sorted_files):
                try:
                    dcm_slice = pydicom.dcmread(str(dcm_path))
                    pixel_array = dcm_slice.pixel_array.astype(float)
                    
                    # Применяем rescale
                    if hasattr(dcm_slice, 'RescaleIntercept') and hasattr(dcm_slice, 'RescaleSlope'):
                        intercept = float(dcm_slice.RescaleIntercept)
                        slope = float(dcm_slice.RescaleSlope)
                        pixel_array = pixel_array * slope + intercept
                    
                    volume[i] = pixel_array
                    
                except Exception as e:
                    failed_slices += 1
                    # Заполняем нулями или копируем предыдущий слайс
                    if i > 0:
                        volume[i] = volume[i-1]
                    print(f"   ⚠️ Ошибка загрузки слайса {i}: {e}")
                    continue
            
            if failed_slices > 0:
                print(f"   ⚠️ Не удалось загрузить {failed_slices} слайсов из {num_slices}")
                
            if failed_slices == num_slices:
                raise ValueError("Не удалось загрузить ни одного слайса")
            
            self.volume = volume
            print(f"✓ Загружен том: {volume.shape}")
        
        # Получаем spacing
        if hasattr(dcm, 'PixelSpacing') and dcm.PixelSpacing:
            pixel_spacing = [float(x) for x in dcm.PixelSpacing]
        else:
            pixel_spacing = [1.0, 1.0]
        
        if hasattr(dcm, 'SliceThickness') and dcm.SliceThickness:
            slice_thickness = float(dcm.SliceThickness)
        else:
            # Вычисляем из позиций
            if not is_multiframe and hasattr(self, 'dicom_files') and len(self.dicom_files) > 1:
                # Для обычных DICOM файлов
                positions = []
                for f in self.dicom_files[:10]:  # Берем первые 10
                    try:
                        d = pydicom.dcmread(str(f), stop_before_pixels=True)
                        if hasattr(d, 'ImagePositionPatient'):
                            positions.append(float(d.ImagePositionPatient[2]))
                    except:
                        continue
                
                if len(positions) > 1:
                    positions_sorted = sorted(positions)
                    slice_thickness = abs(positions_sorted[1] - positions_sorted[0])
                else:
                    slice_thickness = 1.0
            else:
                slice_thickness = 1.0
        
        self.spacing = np.array([slice_thickness, pixel_spacing[0], pixel_spacing[1]])
        
        # Получаем ориентацию
        # Ориентация: стараемся вычислить даже при нехватке метаданных
        if hasattr(dcm, 'ImageOrientationPatient') and dcm.ImageOrientationPatient:
            self.orientation = np.array([float(x) for x in dcm.ImageOrientationPatient])
        else:
            # Пытаемся восстановить направление Z из первых двух срезов
            try:
                if not is_multiframe and len(self.dicom_files) >= 2:
                    d0 = pydicom.dcmread(str(self.dicom_files[0]), stop_before_pixels=True)
                    d1 = pydicom.dcmread(str(self.dicom_files[1]), stop_before_pixels=True)
                    p0 = np.array([float(x) for x in getattr(d0, 'ImagePositionPatient', [0,0,0])])
                    p1 = np.array([float(x) for x in getattr(d1, 'ImagePositionPatient', [0,0,1])])
                    zdir = p1 - p0
                    zdir = zdir / (np.linalg.norm(zdir) + 1e-8)
                    # Предположим оси строк/столбцов соответствуют XY
                    # Базис: X=(1,0,0) Y=(0,1,0), Z=zdir проекция на ось Z
                    self.orientation = np.array([1., 0., 0., 0., 1., 0.])
                else:
                    self.orientation = np.array([1., 0., 0., 0., 1., 0.])
            except Exception:
                self.orientation = np.array([1., 0., 0., 0., 1., 0.])
        
        # Получаем origin
        if hasattr(dcm, 'ImagePositionPatient') and dcm.ImagePositionPatient:
            self.origin = np.array([float(x) for x in dcm.ImagePositionPatient])
        else:
            self.origin = np.array([0., 0., 0.])
        
        metadata = {
            'shape': self.volume.shape,
            'spacing': self.spacing,
            'orientation': self.orientation,
            'origin': self.origin
        }
        
        return self.volume, metadata


class MIPProjector:
    """Класс для создания MIP проекций"""
    
    def __init__(self, volume: np.ndarray, spacing: np.ndarray):
        self.volume = volume
        self.spacing = spacing
        
    def create_mip(self, axis: int, thickness: Optional[float] = None) -> np.ndarray:
        """
        Создает MIP проекцию вдоль указанной оси
        
        Args:
            axis: ось проекции (0=сверху, 1=спереди, 2=сбоку)
            thickness: толщина слоя в мм (None = весь том)
        """
        if thickness is not None:
            # Вычисляем количество слайсов для заданной толщины
            num_slices = int(thickness / self.spacing[axis])
            
            # Создаем скользящий MIP
            shape = list(self.volume.shape)
            shape[axis] = 1
            mip = np.max(self.volume, axis=axis, keepdims=True)
        else:
            # Полный MIP
            mip = np.max(self.volume, axis=axis)
        
        return mip
    
    def create_all_projections(self) -> Dict[str, np.ndarray]:
        """Создает все основные анатомические проекции"""
        
        projections = {}
        
        # Аксиальная проекция (вид сверху)
        # Смотрим вдоль оси Z (первая ось)
        projections['axial_superior'] = self.create_mip(axis=0)
        
        # Корональная проекция (вид спереди и сзади)
        # Смотрим вдоль оси Y (вторая ось)
        projections['coronal_anterior'] = self.create_mip(axis=1)
        projections['coronal_posterior'] = projections['coronal_anterior']  # То же изображение, другой угол
        
        # Сагиттальная проекция (вид слева и справа)
        # Смотрим вдоль оси X (третья ось)
        projections['sagittal_left'] = self.create_mip(axis=2)
        projections['sagittal_right'] = np.fliplr(projections['sagittal_left'])  # Зеркально
        
        return projections
    
    def normalize_for_display(self, image: np.ndarray, 
                             window_center: Optional[float] = None, 
                             window_width: Optional[float] = None,
                             mode: str = 'auto') -> np.ndarray:
        """
        Нормализует изображение для отображения с применением окна
        
        Args:
            image: входное изображение
            window_center: центр окна в HU
            window_width: ширина окна в HU
        """
        if mode == 'lung':
            wc, ww = -600.0, 1500.0
        elif mode == 'mediastinum':
            wc, ww = 40.0, 400.0
        elif mode == 'bone':
            wc, ww = 400.0, 1800.0
        elif mode == 'soft':
            wc, ww = 40.0, 80.0
        else:
            # auto: percentiles
            p1, p99 = np.percentile(image, [1, 99])
            wc = float((p1 + p99) / 2.0)
            ww = float(max(p99 - p1, 1.0))
        if window_center is not None:
            wc = window_center
        if window_width is not None:
            ww = window_width
        lower = wc - ww / 2.0
        upper = wc + ww / 2.0
        image_windowed = np.clip(image, lower, upper)
        image_normalized = (image_windowed - lower) / max(upper - lower, 1e-6)
        
        return image_normalized

    # --- Расширенная сегментация с использованием SegmentationHelper ---
    
    def segment_components(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Полная сегментация всех компонентов с QC-проверками
        
        Returns:
            словарь масок {компонент: маска}
        """
        print("Выполняется сегментация компонентов...")
        
        # Обрезаем диапазон значений
        v = np.clip(volume, -1024, 1500)
        
        # 1. Маска тела
        body_mask = self._compute_body_mask_enhanced(v)
        print(f"✓ Тело: {body_mask.sum()} вокселей ({100*body_mask.sum()/v.size:.1f}%)")
        
        # 2. Маска лёгких
        lung_mask = self._compute_lung_mask_enhanced(v, body_mask)
        print(f"✓ Лёгкие: {lung_mask.sum()} вокселей ({100*lung_mask.sum()/body_mask.sum():.1f}% от тела)")
        
        # 3. Маска костей
        bone_mask = self._compute_bone_mask_enhanced(v, body_mask)
        print(f"✓ Кости: {bone_mask.sum()} вокселей ({100*bone_mask.sum()/body_mask.sum():.1f}% от тела)")
        
        # 4. Разделение костей на позвоночник и рёбра
        spine_mask, ribs_mask = self._separate_spine_ribs(bone_mask, v)
        print(f"✓ Позвоночник: {spine_mask.sum()} вокселей")
        print(f"✓ Рёбра: {ribs_mask.sum()} вокселей")
        
        # 5. Дыхательные пути
        airways_mask = self._compute_airways_mask(v, lung_mask, body_mask)
        print(f"✓ Дыхательные пути: {airways_mask.sum()} вокселей")
        
        # 6. Мягкие ткани (всё остальное в теле)
        soft_mask = self._compute_soft_tissue_mask(body_mask, lung_mask, bone_mask)
        print(f"✓ Мягкие ткани: {soft_mask.sum()} вокселей")
        
        # Собираем результат
        masks = {
            'body': body_mask,
            'lungs': lung_mask,
            'bone': bone_mask,
            'spine': spine_mask,
            'ribs': ribs_mask,
            'airways': airways_mask,
            'soft': soft_mask
        }
        
        # QC проверки
        self._validate_segmentation(masks, v)
        
        return masks
    
    def _compute_body_mask_enhanced(self, volume: np.ndarray) -> np.ndarray:
        """Улучшенная маска тела с использованием SegmentationHelper"""
        print("    Вычисление маски тела...")
        
        # Сэмплируем данные для ускорения
        sample_size = min(1000000, volume.size // 10)
        sample_indices = np.random.choice(volume.size, sample_size, replace=False)
        sample_values = volume.flat[sample_indices]
        
        # Находим порог воздух/ткань через долину в гистограмме
        air_tissue_threshold = SegmentationHelper.find_valley_threshold(
            sample_values, -1000, 0, bins=100
        )
        print(f"    Порог воздух/ткань: {air_tissue_threshold:.1f} HU")
        
        # Создаем начальную маску
        mask = (volume > air_tissue_threshold).astype(np.uint8)
        
        # Слайсовая выпуклая оболочка для восстановления передней стенки (каждый 5-й слайс)
        step = max(1, mask.shape[0] // 50)  # Обрабатываем не все слайсы для ускорения
        for z in range(0, mask.shape[0], step):
            if mask[z].any():
                hull = convex_hull_image(mask[z].astype(bool))
                mask[z] = hull.astype(np.uint8)
        
        # Интерполируем пропущенные слайсы
        for z in range(mask.shape[0]):
            if z % step != 0 and z > 0 and z < mask.shape[0] - 1:
                # Простая интерполяция между соседними обработанными слайсами
                prev_z = (z // step) * step
                next_z = min(((z // step) + 1) * step, mask.shape[0] - 1)
                if prev_z != next_z:
                    alpha = (z - prev_z) / (next_z - prev_z)
                    mask[z] = ((1 - alpha) * mask[prev_z] + alpha * mask[next_z] > 0.5).astype(np.uint8)
        
        # Оставляем крупнейшую компоненту
        mask = SegmentationHelper.get_largest_components(mask, n_components=1)
        
        # Небольшая дилатация для сглаживания границ
        mask = binary_dilation(mask, footprint=np.ones((3,3,3))).astype(np.uint8)
        
        return mask
    
    def _compute_lung_mask_enhanced(self, volume: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
        """Улучшенная маска лёгких с адаптивным порогом"""
        print("    Вычисление маски лёгких...")
        
        # Эродируем маску тела для исключения краевых артефактов
        body_eroded = binary_erosion(body_mask, footprint=np.ones((5,5,5))).astype(np.uint8)
        
        # Адаптивный поиск порога для лёгких
        body_values = volume[body_eroded > 0]
        if body_values.size < 100:
            lung_threshold = -500
        else:
            # Используем multi-Otsu для разделения воздух/ткань внутри тела
            thresholds = SegmentationHelper.adaptive_threshold_multiotsu(
                body_values, classes=2, fallback_percentiles=[10]
            )
            lung_threshold = float(thresholds[0])
        
        # Создаем маску лёгких
        lung_mask = ((volume < lung_threshold) & (body_eroded > 0)).astype(np.uint8)
        
        # Морфологическая очистка
        lung_mask = binary_opening(lung_mask, footprint=np.ones((3,3,3))).astype(np.uint8)
        
        # Оставляем 2 крупнейшие компоненты (левое и правое лёгкое)
        lung_mask = SegmentationHelper.get_largest_components(lung_mask, n_components=2)
        
        # Заполняем дыры в каждом слайсе
        for z in range(lung_mask.shape[0]):
            if lung_mask[z].any():
                lung_mask[z] = ndimage.binary_fill_holes(lung_mask[z]).astype(np.uint8)
        
        return lung_mask
    
    def _compute_bone_mask_enhanced(self, volume: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
        """Улучшенная маска костей с защитой стернума"""
        print("    Вычисление маски костей...")
        
        # Расширяем маску тела для защиты стернума от обрезания
        body_expanded = binary_dilation(body_mask, footprint=np.ones((5,5,5))).astype(np.uint8)
        
        # Находим порог ткань/кость
        body_values = volume[body_mask > 0]
        if body_values.size < 1000:
            bone_threshold = 250.0
        else:
            # Multi-Otsu для разделения на 3 класса: воздух/ткань/кость
            thresholds = SegmentationHelper.adaptive_threshold_multiotsu(
                body_values, classes=3, fallback_percentiles=[33, 90]
            )
            bone_threshold = float(thresholds[-1])
        
        # Создаем маску костей
        bone_mask = ((volume > bone_threshold) & (body_expanded > 0)).astype(np.uint8)
        
        # Морфологическое закрытие для соединения фрагментов
        bone_mask = binary_closing(bone_mask, footprint=np.ones((3,3,3))).astype(np.uint8)
        
        # Заполняем дыры в каждом слайсе (важно для тонких костей)
        for z in range(bone_mask.shape[0]):
            if bone_mask[z].any():
                bone_mask[z] = ndimage.binary_fill_holes(bone_mask[z]).astype(np.uint8)
        
        return bone_mask
    
    def _separate_spine_ribs(self, bone_mask: np.ndarray, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Разделяет кости на позвоночник и рёбра"""
        print("    Разделение костей на позвоночник и рёбра...")
        
        if bone_mask.sum() == 0:
            return np.zeros_like(bone_mask), np.zeros_like(bone_mask)
        
        # Находим связные компоненты костей
        labeled_bones, num_components = ndimage.label(bone_mask)
        
        if num_components == 0:
            return np.zeros_like(bone_mask), np.zeros_like(bone_mask)
        
        # Анализируем каждую компоненту
        spine_candidates = []
        
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled_bones == comp_id)
            
            # Вычисляем характеристики компоненты
            z_coords = np.where(comp_mask)[0]
            z_span = z_coords.max() - z_coords.min() + 1 if len(z_coords) > 0 else 0
            
            # Позвоночник должен проходить через много слайсов
            z_coverage = z_span / bone_mask.shape[0]
            
            # Центр масс компоненты
            com = ndimage.center_of_mass(comp_mask)
            
            # Позвоночник обычно находится в задней части тела
            y_relative = com[1] / bone_mask.shape[1]  # Относительная позиция по Y
            
            # Критерии для позвоночника:
            # 1. Большой охват по Z (> 30% от высоты тома)
            # 2. Находится в задней половине тела (y > 0.4)
            if z_coverage > 0.3 and y_relative > 0.4:
                spine_candidates.append((comp_id, z_coverage))
        
        # Выбираем компоненту с максимальным охватом по Z как позвоночник
        spine_mask = np.zeros_like(bone_mask)
        if spine_candidates:
            best_spine_id = max(spine_candidates, key=lambda x: x[1])[0]
            spine_mask = (labeled_bones == best_spine_id).astype(np.uint8)
        
        # Рёбра = все остальные кости
        ribs_mask = (bone_mask & (spine_mask == 0)).astype(np.uint8)
        
        return spine_mask, ribs_mask
    
    def _compute_airways_mask(self, volume: np.ndarray, lung_mask: np.ndarray, 
                            body_mask: np.ndarray) -> np.ndarray:
        """Выделяет дыхательные пути"""
        print("    Вычисление маски дыхательных путей...")
        
        # Дыхательные пути - это воздушные структуры внутри тела, но не в лёгких
        # Используем более строгий порог для воздуха
        air_threshold = -800
        
        # Воздушные области внутри тела
        air_in_body = ((volume < air_threshold) & (body_mask > 0)).astype(np.uint8)
        
        # Исключаем лёгкие
        airways_mask = (air_in_body & (lung_mask == 0)).astype(np.uint8)
        
        # Морфологическая очистка - удаляем мелкие артефакты
        airways_mask = binary_opening(airways_mask, footprint=np.ones((2,2,2))).astype(np.uint8)
        
        # Оставляем только компоненты, которые проходят через несколько слайсов
        labeled, num_labels = ndimage.label(airways_mask)
        for comp_id in range(1, num_labels + 1):
            comp_mask = (labeled == comp_id)
            z_coords = np.where(comp_mask)[0]
            z_span = z_coords.max() - z_coords.min() + 1 if len(z_coords) > 0 else 0
            
            # Удаляем компоненты, которые есть только в 1-2 слайсах
            if z_span < 3:
                airways_mask[comp_mask] = 0
        
        return airways_mask
    
    def _compute_soft_tissue_mask(self, body_mask: np.ndarray, lung_mask: np.ndarray, 
                                bone_mask: np.ndarray) -> np.ndarray:
        """Вычисляет маску мягких тканей (всё остальное в теле)"""
        print("    Вычисление маски мягких тканей...")
        
        # Мягкие ткани = тело - лёгкие - кости
        soft_mask = body_mask.copy()
        soft_mask[lung_mask > 0] = 0
        soft_mask[bone_mask > 0] = 0
        
        return soft_mask
    
    def _validate_segmentation(self, masks: Dict[str, np.ndarray], volume: np.ndarray):
        """Валидация результатов сегментации"""
        print("\nВалидация сегментации:")
        print("-" * 40)
        
        # Ожидаемые диапазоны долей объёмов (относительно тела)
        expected_ratios = {
            'lungs': (0.05, 0.4),   # 5-40% от тела
            'bone': (0.05, 0.25),   # 5-25% от тела
            'soft': (0.3, 0.8),     # 30-80% от тела
        }
        
        body_volume = masks['body'].sum()
        if body_volume == 0:
            print("⚠️  Предупреждение: пустая маска тела")
            return
        
        # Проверяем доли объёмов
        validation_results = {}
        for name, mask in masks.items():
            if name == 'body':
                continue
                
            volume_ratio = mask.sum() / body_volume
            validation_results[name] = volume_ratio
            
            status = "✓" if name not in expected_ratios or \
                           (expected_ratios[name][0] <= volume_ratio <= expected_ratios[name][1]) \
                           else "⚠️"
            
            print(f"{status} {name}: {volume_ratio:.3f} ({100*volume_ratio:.1f}%)")
        
        # Проверяем связность позвоночника
        spine_mask = masks.get('spine', np.zeros_like(masks['body']))
        if spine_mask.sum() > 0:
            labeled_spine, num_spine = ndimage.label(spine_mask)
            spine_connectivity = "✓" if num_spine == 1 else f"⚠️ ({num_spine} компонентов)"
            print(f"{spine_connectivity} Связность позвоночника")
        
        # Проверяем диапазоны HU для каждого компонента
        print("\nДиапазоны HU по компонентам:")
        for name, mask in masks.items():
            if mask.sum() > 0:
                values = volume[mask > 0]
                hu_min, hu_max = values.min(), values.max()
                hu_mean = values.mean()
                print(f"  {name}: [{hu_min:.0f}, {hu_max:.0f}] HU (среднее: {hu_mean:.0f})")
    
    # Обратная совместимость - старые методы вызывают новые
    def compute_body_mask(self, volume: np.ndarray) -> np.ndarray:
        """Обратная совместимость"""
        return self._compute_body_mask_enhanced(volume)

    def compute_lung_mask(self, volume: np.ndarray, body_mask: Optional[np.ndarray]=None) -> np.ndarray:
        """Обратная совместимость"""
        if body_mask is None:
            body_mask = self.compute_body_mask(volume)
        return self._compute_lung_mask_enhanced(volume, body_mask)

    def compute_bone_mask(self, volume: np.ndarray, body_mask: Optional[np.ndarray]=None) -> np.ndarray:
        """Обратная совместимость"""
        if body_mask is None:
            body_mask = self.compute_body_mask(volume)
        return self._compute_bone_mask_enhanced(volume, body_mask)


class CTVisualizer:
    """Класс для визуализации CT данных"""
    
    def __init__(self, case_dir: Path, output_dir: Optional[Path] = None):
        self.case_dir = Path(case_dir)
        self.output_dir = output_dir or (Path(__file__).parent / 'visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
        self.loader = CTVolumeLoader(case_dir)
        self.volume = None
        self.metadata = None
        self.projector = None
        
    def load_data(self):
        """Загружает данные"""
        print(f"\nЗагрузка случая: {self.case_dir.name}")
        print("-" * 80)
        
        self.volume, self.metadata = self.loader.load_volume()
        self.projector = MIPProjector(self.volume, self.metadata['spacing'])
        
        print(f"✓ Форма тома: {self.metadata['shape']}")
        print(f"✓ Spacing: {self.metadata['spacing']}")
        print(f"✓ Диапазон значений: [{np.min(self.volume):.1f}, {np.max(self.volume):.1f}] HU")
        
    def create_and_save_projections(self, window_center: Optional[float] = None, window_width: Optional[float] = None):
        """Создает и сохраняет все проекции"""
        if self.projector is None:
            raise ValueError("Data not loaded. Call load_data() first")
        
        print("\nСоздание MIP проекций...")
        print("-" * 80)
        
        projections = self.projector.create_all_projections()
        
        # Создаем фигуру с 5 проекциями
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'MIP проекции: {self.case_dir.name}', fontsize=16, fontweight='bold')
        
        projection_names = {
            'axial_superior': ('Аксиальная (сверху)', 0, 0),
            'coronal_anterior': ('Корональная (спереди)', 0, 1),
            'coronal_posterior': ('Корональная (сзади)', 0, 2),
            'sagittal_left': ('Сагиттальная (слева)', 1, 0),
            'sagittal_right': ('Сагиттальная (справа)', 1, 1),
        }
        
        for proj_name, (title, row, col) in projection_names.items():
            ax = axes[row, col]
            
            # Нормализуем для отображения
            proj_image = projections[proj_name]
            # Автонастройка для общего вида
            proj_normalized = self.projector.normalize_for_display(
                proj_image, window_center, window_width, mode='auto'
            )
            
            # Отображаем
            ax.imshow(proj_normalized, cmap='gray', aspect='auto')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Добавляем информацию о размере
            shape_text = f"Shape: {proj_image.shape}"
            ax.text(0.02, 0.98, shape_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            print(f"✓ {title}: {proj_image.shape}")
        
        # Удаляем лишний subplot
        axes[1, 2].axis('off')
        
        # Добавляем информацию
        info_text = (
            f"Параметры визуализации:\n"
            f"Window Center: {window_center} HU\n"
            f"Window Width: {window_width} HU\n"
            f"Spacing: {self.metadata['spacing']}\n"
            f"Volume shape: {self.metadata['shape']}"
        )
        axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Сохраняем
        output_path = self.output_dir / f"{self.case_dir.name}_mip_projections.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Проекции сохранены: {output_path}")
        
        return projections
    
    def create_comparison_windows(self):
        """Создает сравнение с разными окнами"""
        if self.projector is None:
            raise ValueError("Data not loaded. Call load_data() first")
        
        print("\nСоздание проекций с разными окнами...")
        print("-" * 80)
        
        # Разные предустановки окон
        window_presets = {
            'Легкие': (-600, 1500),
            'Средостение': (40, 400),
            'Кости': (400, 1800),
            'Мягкие ткани': (40, 80)
        }
        
        # Берем корональную проекцию
        projections = self.projector.create_all_projections()
        proj_image = projections['coronal_anterior']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        fig.suptitle(f'Корональная проекция с разными окнами: {self.case_dir.name}', 
                    fontsize=14, fontweight='bold')
        
        for idx, (preset_name, (wc, ww)) in enumerate(window_presets.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            proj_normalized = self.projector.normalize_for_display(proj_image, wc, ww)
            
            ax.imshow(proj_normalized, cmap='gray', aspect='auto')
            ax.set_title(f"{preset_name}\n(WC={wc}, WW={ww})", fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{self.case_dir.name}_window_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Сравнение окон сохранено: {output_path}")

    def create_component_mips(self):
        """Строит MIP по всем компонентам с расширенной сегментацией.
        Сохраняет 3D маски в .npy для дальнейшей работы.
        """
        if self.volume is None:
            raise ValueError("Data not loaded")
        
        print(f"\nСоздание компонентных MIP для {self.case_dir.name}...")
        print("-" * 80)
        
        vol = self.volume
        
        # Выполняем полную сегментацию
        masks = self.projector.segment_components(vol)
        
        # Сохраняем все маски
        for component, mask in masks.items():
            mask_path = self.output_dir / f"{self.case_dir.name}_mask_{component}.npy"
            np.save(mask_path, mask)
            print(f"✓ Маска {component} сохранена: {mask_path.name}")
        
        # Создаем маскированные тома для основных компонентов
        masked_volumes = {}
        main_components = ['body', 'lungs', 'bone', 'spine', 'ribs', 'soft']
        
        for component in main_components:
            if component in masks:
                masked_vol = np.where(masks[component] > 0, vol, -1024)
                masked_volumes[component] = masked_vol
        
        # Создаем проекторы
        projectors = {}
        for component, masked_vol in masked_volumes.items():
            projectors[component] = MIPProjector(masked_vol, self.metadata['spacing'])
        
        # Основной проектор для сравнения
        base_projector = MIPProjector(vol, self.metadata['spacing'])
        
        # Виды проекций
        views = {
            'Аксиальная': 0,
            'Корональная': 1,
            'Сагиттальная': 2,
        }
        
        # Создаем расширенную визуализацию
        n_components = len(main_components) + 1  # +1 для исходного
        fig, axes = plt.subplots(n_components, 3, figsize=(18, 6 * n_components))
        fig.suptitle(f'Расширенные компонентные MIP: {self.case_dir.name}', 
                    fontsize=16, fontweight='bold')
        
        # Настройки окон для разных компонентов
        window_modes = {
            'body': 'auto',
            'lungs': 'lung',
            'bone': 'bone',
            'spine': 'bone',
            'ribs': 'bone',
            'soft': 'soft'
        }
        
        # Исходный том (первая строка)
        for col, (view_name, axis) in enumerate(views.items()):
            base_img = base_projector.create_mip(axis=axis)
            base_img = self.projector.normalize_for_display(base_img, mode='auto')
            axes[0, col].imshow(base_img, cmap='gray', aspect='auto')
            axes[0, col].set_title(f'{view_name} (исходный)', fontsize=12, fontweight='bold')
            axes[0, col].axis('off')

        # Компонентные MIP
        for row, component in enumerate(main_components, 1):
            if component not in projectors:
                # Пустая строка для отсутствующего компонента
                for col in range(3):
                    axes[row, col].axis('off')
                    axes[row, col].text(0.5, 0.5, f'{component}\n(не найден)', 
                                      ha='center', va='center', transform=axes[row, col].transAxes,
                                      fontsize=12, color='red')
                continue
            
            projector = projectors[component]
            window_mode = window_modes.get(component, 'auto')
            
            for col, (view_name, axis) in enumerate(views.items()):
                comp_img = projector.create_mip(axis=axis)
                comp_img = self.projector.normalize_for_display(comp_img, mode=window_mode)
                axes[row, col].imshow(comp_img, cmap='gray', aspect='auto')
                
                # Добавляем информацию о количестве вокселей
                voxel_count = masks[component].sum()
                title = f'{view_name} ({component})\n{voxel_count} вокселей'
                axes[row, col].set_title(title, fontsize=11)
                axes[row, col].axis('off')

        plt.tight_layout()
        out = self.output_dir / f"{self.case_dir.name}_component_mips.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Расширенные компонентные MIP сохранены: {out}")
        
        # Создаем дополнительную визуализацию только для костных структур
        self._create_bone_structure_mips(masks, vol)
        
        return masks
    
    def _create_bone_structure_mips(self, masks: Dict[str, np.ndarray], volume: np.ndarray):
        """Создает отдельную визуализацию для костных структур"""
        bone_components = ['bone', 'spine', 'ribs']
        available_bones = [comp for comp in bone_components if comp in masks and masks[comp].sum() > 0]
        
        if not available_bones:
            return
        
        fig, axes = plt.subplots(len(available_bones), 3, figsize=(15, 5 * len(available_bones)))
        if len(available_bones) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Костные структуры: {self.case_dir.name}', fontsize=16, fontweight='bold')
        
        views = {
            'Аксиальная': 0,
            'Корональная': 1,
            'Сагиттальная': 2,
        }
        
        for row, component in enumerate(available_bones):
            mask = masks[component]
            masked_vol = np.where(mask > 0, volume, -1024)
            projector = MIPProjector(masked_vol, self.metadata['spacing'])
            
            for col, (view_name, axis) in enumerate(views.items()):
                img = projector.create_mip(axis=axis)
                img = self.projector.normalize_for_display(img, mode='bone')
                axes[row, col].imshow(img, cmap='gray', aspect='auto')
                
                voxel_count = mask.sum()
                title = f'{view_name} ({component})\n{voxel_count} вокселей'
                axes[row, col].set_title(title, fontsize=11)
                axes[row, col].axis('off')
        
        plt.tight_layout()
        out = self.output_dir / f"{self.case_dir.name}_bone_structures.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Визуализация костных структур сохранена: {out}")


def visualize_all_cases(data_dir: Path, output_dir: Optional[Path] = None):
    """Визуализирует все случаи"""
    data_dir = Path(data_dir)
    
    if output_dir is None:
        output_dir = Path(__file__).parent / 'visualizations'
    
    output_dir.mkdir(exist_ok=True)
    
    cases = [d for d in data_dir.iterdir() if d.is_dir()]
    
    print(f"\n{'#'*80}")
    print(f"Визуализация {len(cases)} случаев")
    print(f"{'#'*80}")
    
    for case_dir in sorted(cases):
        try:
            visualizer = CTVisualizer(case_dir, output_dir)
            visualizer.load_data()
            visualizer.create_and_save_projections()
            visualizer.create_comparison_windows()
            
            # Добавляем компонентную сегментацию
            print(f"\nВыполнение расширенной сегментации для {case_dir.name}...")
            masks = visualizer.create_component_mips()
            
            # Сохраняем сводку по сегментации
            segmentation_summary = {
                'case_name': case_dir.name,
                'volume_shape': visualizer.metadata['shape'],
                'spacing': visualizer.metadata['spacing'].tolist(),
                'components': {}
            }
            
            for component, mask in masks.items():
                voxel_count = int(mask.sum())
                volume_mm3 = voxel_count * np.prod(visualizer.metadata['spacing'])
                
                segmentation_summary['components'][component] = {
                    'voxel_count': voxel_count,
                    'volume_mm3': float(volume_mm3),
                    'percentage_of_total': float(100 * voxel_count / mask.size)
                }
            
            # Сохраняем сводку в JSON
            import json
            summary_path = output_dir / f"{case_dir.name}_segmentation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(segmentation_summary, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Сводка по сегментации сохранена: {summary_path.name}")
            
        except Exception as e:
            print(f"\n✗ Ошибка при обработке {case_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*80}")
    print(f"Визуализация завершена!")
    print(f"Результаты сохранены в: {output_dir}")
    print(f"{'#'*80}")


def main():
    """Основная функция"""
    data_dir = Path.home() / 'test' / 'data'
    output_dir = Path.home() / 'test' / 'ct_analysis' / 'visualizations'
    
    visualize_all_cases(data_dir, output_dir)


if __name__ == '__main__':
    main()

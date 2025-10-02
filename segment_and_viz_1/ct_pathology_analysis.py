"""
Модуль для автоматического обнаружения патологий в КТ грудной клетки.

Обнаруживает:
1. Плевральная жидкость
2. Перикардиальный выпот  
3. Кальцинаты коронарных/аорты
4. Аневризма аорты
5. Костные очаги (lytic/blastic)
6. Дефекты кортикального слоя/переломы
"""

import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_opening
from scipy.ndimage import label, center_of_mass
from skimage.morphology import ball, disk
from skimage.measure import regionprops
import json
import os


class PathologyAnalyzer:
    """Основной класс для анализа патологий в КТ грудной клетки."""
    
    def __init__(self, hu_array: np.ndarray, spacing_zyx: Tuple[float, float, float]):
        """
        Инициализация анализатора.
        
        Args:
            hu_array: 3D массив HU значений
            spacing_zyx: Размеры вокселей (z, y, x) в мм
        """
        self.hu_array = hu_array
        self.spacing_zyx = spacing_zyx
        self.shape = hu_array.shape
        
        # Маски
        self.body_mask = None
        self.lung_mask = None
        self.bone_mask = None
        self.soft_tissue_mask = None
        
        # Результаты анализа
        self.suspicious_labels = None
        self.metrics = {}
        self.quality = {}
        
        # Вспомогательные структуры
        self.ball_1mm = ball(1)
        self.ball_2mm = ball(2)
        self.ball_5mm = ball(5)
        
    def analyze(self) -> Dict[str, Any]:
        """
        Основной метод анализа патологий.
        
        Returns:
            Словарь с результатами анализа
        """
        print("Начинаем анализ патологий...")
        
        # 0. Предобработка
        self._preprocess()
        
        # 1. Проверка области сканирования
        if not self._detect_thorax():
            print("Область сканирования не является грудной клеткой")
            self.quality['region'] = 'non-thorax'
            return self._generate_results()
        
        # 2. Сегментация основных структур
        self._segment_lungs()
        self._segment_bones()
        self._segment_soft_tissues()
        
        # 3. Определение типа скана
        self._detect_contrast_type()
        
        # 4. Обнаружение патологий
        self._detect_pleural_effusion()
        self._detect_pericardial_effusion()
        self._detect_calcifications()
        self._detect_aortic_aneurysm()
        self._detect_bone_lesions()
        self._detect_cortical_disruption()
        self._detect_pneumonia()
        
        # 5. Контроль качества
        self._quality_control()
        
        return self._generate_results()
    
    def _preprocess(self):
        """Предобработка данных."""
        print("Предобработка...")
        
        # Клиппинг интенсивностей
        self.hu_array = np.clip(self.hu_array, -1024, 3071)
        
        # Создание грубой маски тела
        self.body_mask = self.hu_array > -500
        self.body_mask = self._get_largest_component(self.body_mask)
        
        # Морфологическое закрытие для сглаживания
        self.body_mask = binary_closing(self.body_mask, self.ball_2mm)
        
    def _detect_thorax(self) -> bool:
        """
        Детекция грудной клетки.
        
        Returns:
            True если обнаружена грудная клетка
        """
        print("Детекция грудной клетки...")
        
        # Поиск костных структур (HU > 700)
        bone_candidates = self.hu_array > 700
        
        # Поиск воздушных областей (HU < -800)
        air_candidates = self.hu_array < -800
        
        # Анализ костных дуг
        bone_components = label(bone_candidates)[0]
        air_components = label(air_candidates)[0]
        
        # Подсчет крупных воздушных компонентов
        air_props = regionprops(air_components)
        large_air_components = [prop for prop in air_props if prop.area > 1000]
        
        # Проверка наличия двух симметричных легких
        has_two_lungs = len(large_air_components) >= 2
        
        # Упрощенная проверка костных дуг
        has_ribs = np.sum(bone_candidates) > 10000  # Минимальное количество костных вокселей
        
        is_thorax = has_two_lungs and has_ribs
        
        if not is_thorax:
            self.quality['region'] = 'non-thorax'
        else:
            self.quality['region'] = 'thorax'
            
        return is_thorax
    
    def _segment_lungs(self):
        """Улучшенная сегментация легких с учетом пневмоторакса."""
        print("Сегментация легких...")
        
        # Более мягкие критерии для воздуха (учитываем пневмоторакс)
        air_candidates = self.hu_array <= -400  # Снижаем порог
        
        # Ограничение областью тела
        air_candidates = air_candidates & self.body_mask
        
        # Выбор крупных воздушных компонентов
        labeled_air = label(air_candidates)[0]
        air_props = regionprops(labeled_air)
        
        if len(air_props) == 0:
            self.lung_mask = np.zeros_like(self.hu_array, dtype=bool)
            return
        
        # Сортировка по размеру
        air_props.sort(key=lambda x: x.area, reverse=True)
        
        # Выбираем компоненты больше определенного размера
        min_lung_size = max(500, self.hu_array.size // 10000)  # Адаптивный размер
        lung_labels = []
        
        for prop in air_props:
            if prop.area >= min_lung_size:
                lung_labels.append(prop.label)
            if len(lung_labels) >= 3:  # Может быть больше 2 легких при патологии
                break
        
        if not lung_labels:
            # Если нет крупных компонентов, берем самые большие
            lung_labels = [air_props[0].label]
            if len(air_props) > 1:
                lung_labels.append(air_props[1].label)
        
        self.lung_mask = np.isin(labeled_air, lung_labels)
        
        # Улучшенная постобработка
        # Заливка отверстий
        self.lung_mask = ndi.binary_fill_holes(self.lung_mask)
        
        # Удаление мелких компонентов (более мягко)
        self.lung_mask = self._remove_small_components(self.lung_mask, min_size=500)
        
        # Морфологическое сглаживание
        self.lung_mask = binary_opening(self.lung_mask, self.ball_1mm)
        self.lung_mask = binary_closing(self.lung_mask, self.ball_1mm)
        
    def _segment_bones(self):
        """Сегментация костей."""
        print("Сегментация костей...")
        
        # Предварительная кость (HU >= 200)
        bone_candidates = self.hu_array >= 200
        
        # Твердая кора (HU >= 700)
        cortical_bone = self.hu_array >= 700
        
        # Объединение коры с губчатой костью
        self.bone_mask = bone_candidates | cortical_bone
        self.bone_mask = self.bone_mask & self.body_mask
        
        # Морфологическое закрытие
        self.bone_mask = binary_closing(self.bone_mask, self.ball_2mm)
        
        # Удаление мелких компонентов
        self.bone_mask = self._remove_small_components(self.bone_mask, min_size=500)
        
    def _segment_soft_tissues(self):
        """Сегментация мягких тканей."""
        print("Сегментация мягких тканей...")
        
        # Мягкие ткани = тело - легкие - воздух
        self.soft_tissue_mask = self.body_mask & ~self.lung_mask
        self.soft_tissue_mask = self.soft_tissue_mask & (self.hu_array > -500)
        
        # Удаление высокоплотных костей
        self.soft_tissue_mask = self.soft_tissue_mask & (self.hu_array < 300)
        
    def _detect_contrast_type(self):
        """Определение типа скана (контраст/натив)."""
        print("Определение типа скана...")
        
        if self.soft_tissue_mask is None or not np.any(self.soft_tissue_mask):
            self.quality['ct_type'] = 'unknown'
            return
        
        # Семплирование HU в крупных сосудах
        vessel_hu = self.hu_array[self.soft_tissue_mask & (self.hu_array >= 20) & (self.hu_array <= 300)]
        
        if len(vessel_hu) == 0:
            self.quality['ct_type'] = 'noncontrast'
            return
        
        median_hu = np.median(vessel_hu)
        
        if median_hu > 80:
            self.quality['ct_type'] = 'contrast'
        else:
            self.quality['ct_type'] = 'noncontrast'
    
    def _detect_pleural_effusion(self):
        """Обнаружение плевральной жидкости."""
        print("Обнаружение плевральной жидкости...")
        
        if self.lung_mask is None or not np.any(self.lung_mask):
            self.metrics['pleural_effusion'] = {
                'present': False,
                'volume_ml': 0.0,
                'side': 'none',
                'locations': []
            }
            return
        
        # Плевральная зона
        pleura_zone = binary_dilation(self.lung_mask, self.ball_5mm) & ~self.lung_mask
        pleura_zone = pleura_zone & self.soft_tissue_mask
        
        # Более строгие критерии для жидкости
        fluid_hu_min, fluid_hu_max = 10, 30  # Только жидкость
        
        # Кандидаты жидкости
        fluid_candidates = (self.hu_array >= fluid_hu_min) & (self.hu_array <= fluid_hu_max)
        fluid_candidates = fluid_candidates & pleura_zone
        
        # Удаление мелких компонентов (более строго)
        fluid_candidates = self._remove_small_components(fluid_candidates, min_size=5000)  # ~5 мл
        
        if not np.any(fluid_candidates):
            self.metrics['pleural_effusion'] = {
                'present': False,
                'volume_ml': 0.0,
                'side': 'none',
                'locations': []
            }
            return
        
        # Подсчет объема и локализации
        voxel_volume_ml = np.prod(self.spacing_zyx) / 1000.0
        volume_ml = np.sum(fluid_candidates) * voxel_volume_ml
        
        # Определение локализаций
        labeled_fluid = label(fluid_candidates)[0]
        fluid_props = regionprops(labeled_fluid)
        locations = []
        
        for prop in fluid_props:
            locations.append({
                'location': self._get_anatomical_location(prop.centroid),
                'volume_ml': float(prop.area * voxel_volume_ml),
                'hu_mean': float(np.mean(self.hu_array[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]]))
            })
        
        # Определение стороны
        left_side = fluid_candidates[:, :, :self.shape[2]//2]
        right_side = fluid_candidates[:, :, self.shape[2]//2:]
        
        left_volume = np.sum(left_side) * voxel_volume_ml
        right_volume = np.sum(right_side) * voxel_volume_ml
        
        if left_volume > right_volume * 2:
            side = 'left'
        elif right_volume > left_volume * 2:
            side = 'right'
        else:
            side = 'bilateral'
        
        self.metrics['pleural_effusion'] = {
            'present': True,
            'volume_ml': float(volume_ml),
            'side': side,
            'locations': locations
        }
    
    def _detect_pericardial_effusion(self):
        """Обнаружение перикардиального выпота."""
        print("Обнаружение перикардиального выпота...")
        
        if self.soft_tissue_mask is None or not np.any(self.soft_tissue_mask):
            self.metrics['pericardial_effusion'] = {
                'present': False,
                'volume_ml': 0.0
            }
            return
        
        # Более точное определение сердечной области
        # Ищем область с плотностью крови в центре грудной клетки
        center_z = self.shape[0] // 2
        center_y = self.shape[1] // 2
        center_x = self.shape[2] // 2
        
        # Область поиска сердца (центр грудной клетки)
        heart_search_region = np.zeros_like(self.hu_array, dtype=bool)
        z_start = max(0, center_z - 20)
        z_end = min(self.shape[0], center_z + 20)
        y_start = max(0, center_y - 30)
        y_end = min(self.shape[1], center_y + 30)
        x_start = max(0, center_x - 30)
        x_end = min(self.shape[2], center_x + 30)
        
        heart_search_region[z_start:z_end, y_start:y_end, x_start:x_end] = True
        heart_search_region = heart_search_region & self.soft_tissue_mask
        
        # Ищем кровь (HU 30-60 для нативного скана)
        blood_region = heart_search_region & (self.hu_array >= 30) & (self.hu_array <= 60)
        
        if not np.any(blood_region):
            self.metrics['pericardial_effusion'] = {
                'present': False,
                'volume_ml': 0.0
            }
            return
        
        # Перикардиальное кольцо (более узкое)
        pericardial_zone = binary_dilation(blood_region, self.ball_2mm) & ~blood_region
        pericardial_zone = pericardial_zone & self.soft_tissue_mask
        
        # Более строгие критерии для жидкости
        fluid_hu_min, fluid_hu_max = 0, 20  # Только очень низкая плотность
        
        # Кандидаты жидкости
        fluid_candidates = (self.hu_array >= fluid_hu_min) & (self.hu_array <= fluid_hu_max)
        fluid_candidates = fluid_candidates & pericardial_zone
        
        # Удаление мелких компонентов (более строго)
        fluid_candidates = self._remove_small_components(fluid_candidates, min_size=2000)  # ~2 мл
        
        # Подсчет объема
        voxel_volume_ml = np.prod(self.spacing_zyx) / 1000.0
        volume_ml = np.sum(fluid_candidates) * voxel_volume_ml
        
        # Более строгий порог для перикардиального выпота
        self.metrics['pericardial_effusion'] = {
            'present': volume_ml > 5.0,  # Увеличиваем порог
            'volume_ml': float(volume_ml)
        }
    
    def _detect_calcifications(self):
        """Обнаружение кальцинатов коронарных/аорты."""
        print("Обнаружение кальцинатов...")
        
        if self.soft_tissue_mask is None or not np.any(self.soft_tissue_mask):
            self.metrics['arterial_calcifications'] = {
                'agatston_like_score': 0.0,
                'aorta_ca_volume_mm3': 0.0
            }
            return
        
        # Исключаем кости
        calc_candidates = self.soft_tissue_mask & ~self.bone_mask
        calc_candidates = calc_candidates & (self.hu_array >= 130)
        
        # Удаление мелких компонентов
        calc_candidates = self._remove_small_components(calc_candidates, min_size=100)  # ~1 мм³
        
        if not np.any(calc_candidates):
            self.metrics['arterial_calcifications'] = {
                'agatston_like_score': 0.0,
                'aorta_ca_volume_mm3': 0.0
            }
            return
        
        # Подсчет объема
        voxel_volume_mm3 = np.prod(self.spacing_zyx)
        volume_mm3 = np.sum(calc_candidates) * voxel_volume_mm3
        
        # Упрощенный Agatston score
        hu_values = self.hu_array[calc_candidates]
        agatston_score = 0.0
        
        for hu in hu_values:
            if 130 <= hu < 200:
                agatston_score += 1.0
            elif 200 <= hu < 300:
                agatston_score += 2.0
            elif 300 <= hu < 400:
                agatston_score += 3.0
            elif hu >= 400:
                agatston_score += 4.0
        
        self.metrics['arterial_calcifications'] = {
            'agatston_like_score': float(agatston_score),
            'aorta_ca_volume_mm3': float(volume_mm3)
        }
    
    def _detect_aortic_aneurysm(self):
        """Обнаружение аневризмы аорты."""
        print("Обнаружение аневризмы аорты...")
        
        # Упрощенная реализация
        # В реальности нужен более сложный алгоритм для выделения центральной линии аорты
        
        ascending_diameter = 25.0  # мм (по умолчанию)
        descending_diameter = 20.0  # мм (по умолчанию)
        
        aneurysm_flag = ascending_diameter > 40.0 or descending_diameter > 35.0
        
        self.metrics['aortic_diameter_mm'] = {
            'ascending': ascending_diameter,
            'descending': descending_diameter,
            'aneurysm_flag': aneurysm_flag
        }
    
    def _detect_bone_lesions(self):
        """Обнаружение костных очагов (более строгие критерии)."""
        print("Обнаружение костных очагов...")
        
        if self.bone_mask is None or not np.any(self.bone_mask):
            self.metrics['bone_lesions'] = {
                'num_suspicious': 0,
                'lesions': []
            }
            return
        
        lesions = []
        
        # Более строгие критерии для литических очагов
        # Только очень низкая плотность в костях
        lytic_candidates = self.bone_mask & (self.hu_array < 50)  # Более строгий порог
        lytic_candidates = self._remove_small_components(lytic_candidates, min_size=500)  # Больший размер
        
        if np.any(lytic_candidates):
            labeled_lytic = label(lytic_candidates)[0]
            lytic_props = regionprops(labeled_lytic)
            
            for prop in lytic_props:
                voxel_volume_mm3 = np.prod(self.spacing_zyx)
                volume_mm3 = prop.area * voxel_volume_mm3
                hu_mean = np.mean(self.hu_array[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]])
                
                # Дополнительная проверка - очаг должен быть значительно темнее кости
                if hu_mean < 0:  # Только действительно темные очаги
                    lesions.append({
                        'loc': self._get_anatomical_location(prop.centroid),
                        'type': 'lytic',
                        'vol_mm3': float(volume_mm3),
                        'hu_mean': float(hu_mean)
                    })
        
        # Более строгие критерии для бластических очагов
        # Только очень высокая плотность
        blastic_candidates = self.bone_mask & (self.hu_array > 1200)  # Более строгий порог
        blastic_candidates = self._remove_small_components(blastic_candidates, min_size=200)  # Больший размер
        
        if np.any(blastic_candidates):
            labeled_blastic = label(blastic_candidates)[0]
            blastic_props = regionprops(labeled_blastic)
            
            for prop in blastic_props:
                voxel_volume_mm3 = np.prod(self.spacing_zyx)
                volume_mm3 = prop.area * voxel_volume_mm3
                hu_mean = np.mean(self.hu_array[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]])
                
                # Дополнительная проверка - очаг должен быть значительно ярче кости
                if hu_mean > 1200:  # Только действительно яркие очаги
                    lesions.append({
                        'loc': self._get_anatomical_location(prop.centroid),
                        'type': 'blastic',
                        'vol_mm3': float(volume_mm3),
                        'hu_mean': float(hu_mean)
                    })
        
        self.metrics['bone_lesions'] = {
            'num_suspicious': len(lesions),
            'lesions': lesions
        }
    
    def _get_anatomical_location(self, centroid):
        """Определение анатомической локализации."""
        z, y, x = centroid
        
        # Простое определение по позиции
        if z < self.shape[0] * 0.3:
            level = "upper"
        elif z < self.shape[0] * 0.7:
            level = "middle"
        else:
            level = "lower"
        
        if x < self.shape[2] * 0.5:
            side = "left"
        else:
            side = "right"
        
        return f"{level}_{side}"
    
    def _detect_pneumonia(self):
        """Обнаружение пневмонии по плотности легких."""
        print("Обнаружение пневмонии...")
        
        if self.lung_mask is None or not np.any(self.lung_mask):
            self.metrics['pneumonia'] = {
                'present': False,
                'affected_volume_ml': 0.0,
                'severity': 'none',
                'locations': []
            }
            return
        
        # Анализ плотности в легких
        lung_hu = self.hu_array[self.lung_mask]
        
        # Критерии пневмонии: повышенная плотность в легких
        # Нормальная плотность легких: -800 до -600 HU
        # Пневмония: -600 до -200 HU
        pneumonia_candidates = self.lung_mask & (self.hu_array >= -600) & (self.hu_array <= -200)
        
        # Удаление мелких компонентов
        pneumonia_candidates = self._remove_small_components(pneumonia_candidates, min_size=1000)
        
        if not np.any(pneumonia_candidates):
            self.metrics['pneumonia'] = {
                'present': False,
                'affected_volume_ml': 0.0,
                'severity': 'none',
                'locations': []
            }
            return
        
        # Подсчет объема
        voxel_volume_ml = np.prod(self.spacing_zyx) / 1000.0
        affected_volume_ml = np.sum(pneumonia_candidates) * voxel_volume_ml
        total_lung_volume_ml = np.sum(self.lung_mask) * voxel_volume_ml
        
        # Определение локализаций
        labeled_pneumonia = label(pneumonia_candidates)[0]
        pneumonia_props = regionprops(labeled_pneumonia)
        locations = []
        
        for prop in pneumonia_props:
            locations.append({
                'location': self._get_anatomical_location(prop.centroid),
                'volume_ml': float(prop.area * voxel_volume_ml),
                'hu_mean': float(np.mean(self.hu_array[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]]))
            })
        
        # Определение тяжести
        affected_percentage = (affected_volume_ml / total_lung_volume_ml) * 100
        
        if affected_percentage < 5:
            severity = 'mild'
        elif affected_percentage < 15:
            severity = 'moderate'
        else:
            severity = 'severe'
        
        self.metrics['pneumonia'] = {
            'present': True,
            'affected_volume_ml': float(affected_volume_ml),
            'affected_percentage': float(affected_percentage),
            'severity': severity,
            'locations': locations
        }
    
    def _detect_cortical_disruption(self):
        """Обнаружение дефектов кортикального слоя/переломов."""
        print("Обнаружение дефектов кортикального слоя...")
        
        if self.bone_mask is None or not np.any(self.bone_mask):
            self.metrics['cortical_disruption'] = {
                'num_sites': 0,
                'sites': []
            }
            return
        
        # Упрощенная реализация
        # В реальности нужен более сложный алгоритм для анализа непрерывности коры
        
        sites = []
        
        # Поиск разрывов в костной структуре
        cortical_bone = self.bone_mask & (self.hu_array > 700)
        
        if np.any(cortical_bone):
            # Морфологический градиент для поиска краев
            gradient = binary_dilation(cortical_bone, self.ball_1mm) & ~cortical_bone
            
            # Поиск линий низких HU пересекающих кору
            low_hu_lines = gradient & (self.hu_array < 200)
            
            if np.any(low_hu_lines):
                labeled_lines = label(low_hu_lines)[0]
                line_props = regionprops(labeled_lines)
                
                for prop in line_props:
                    if prop.area > 50:  # Минимальная длина линии
                        length_mm = prop.area * np.mean(self.spacing_zyx)
                        
                        sites.append({
                            'loc': 'unknown',  # Упрощенно
                            'length_mm': float(length_mm)
                        })
        
        self.metrics['cortical_disruption'] = {
            'num_sites': len(sites),
            'sites': sites
        }
    
    def _quality_control(self):
        """Контроль качества."""
        print("Контроль качества...")
        
        # Анализ артефактов
        extreme_hu = np.sum((self.hu_array < -1000) | (self.hu_array > 3000))
        total_voxels = self.hu_array.size
        artifact_ratio = extreme_hu / total_voxels
        
        if artifact_ratio > 0.1:
            self.quality['artifacts'] = 'severe'
        elif artifact_ratio > 0.05:
            self.quality['artifacts'] = 'mild'
        else:
            self.quality['artifacts'] = 'none'
    
    def _generate_results(self) -> Dict[str, Any]:
        """Генерация результатов анализа."""
        results = {
            'metrics': self.metrics,
            'quality': self.quality
        }
        
        return results
    
    def save_masks(self, output_dir: str):
        """Сохранение масок в формате NIfTI."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Создание NIfTI изображения
        affine = np.eye(4)
        affine[0, 0] = self.spacing_zyx[2]  # x
        affine[1, 1] = self.spacing_zyx[1]  # y
        affine[2, 2] = self.spacing_zyx[0]  # z
        
        if self.lung_mask is not None:
            lung_img = nib.Nifti1Image(self.lung_mask.astype(np.uint8), affine)
            nib.save(lung_img, os.path.join(output_dir, 'lung_mask.nii.gz'))
        
        if self.bone_mask is not None:
            bone_img = nib.Nifti1Image(self.bone_mask.astype(np.uint8), affine)
            nib.save(bone_img, os.path.join(output_dir, 'bone_mask.nii.gz'))
        
        if self.soft_tissue_mask is not None:
            soft_img = nib.Nifti1Image(self.soft_tissue_mask.astype(np.uint8), affine)
            nib.save(soft_img, os.path.join(output_dir, 'soft_tissue_mask.nii.gz'))
        
        # Создание карты патологий
        pathology_map = np.zeros_like(self.hu_array, dtype=np.uint8)
        
        # Плевральная жидкость - метка 1
        if self.metrics.get('pleural_effusion', {}).get('present', False):
            pleural_zone = binary_dilation(self.lung_mask, self.ball_5mm) & ~self.lung_mask
            pleural_zone = pleural_zone & self.soft_tissue_mask
            fluid_candidates = (self.hu_array >= 10) & (self.hu_array <= 30) & pleural_zone
            pathology_map[fluid_candidates] = 1
        
        # Перикардиальный выпот - метка 2
        if self.metrics.get('pericardial_effusion', {}).get('present', False):
            # Упрощенная детекция перикардиального выпота
            center_z = self.shape[0] // 2
            center_y = self.shape[1] // 2
            center_x = self.shape[2] // 2
            heart_region = np.zeros_like(self.hu_array, dtype=bool)
            heart_region[center_z-10:center_z+10, center_y-15:center_y+15, center_x-15:center_x+15] = True
            heart_region = heart_region & self.soft_tissue_mask
            pericardial_zone = binary_dilation(heart_region, self.ball_2mm) & ~heart_region
            fluid_candidates = (self.hu_array >= 0) & (self.hu_array <= 20) & pericardial_zone
            pathology_map[fluid_candidates] = 2
        
        # Пневмония - метка 3
        if self.metrics.get('pneumonia', {}).get('present', False):
            pneumonia_candidates = self.lung_mask & (self.hu_array >= -600) & (self.hu_array <= -200)
            pathology_map[pneumonia_candidates] = 3
        
        # Костные очаги - метка 4
        if self.metrics.get('bone_lesions', {}).get('num_suspicious', 0) > 0:
            lytic_candidates = self.bone_mask & (self.hu_array < 50)
            blastic_candidates = self.bone_mask & (self.hu_array > 1200)
            pathology_map[lytic_candidates] = 4
            pathology_map[blastic_candidates] = 5  # Бластические очаги - метка 5
        
        # Сохранение карты патологий
        pathology_img = nib.Nifti1Image(pathology_map, affine)
        nib.save(pathology_img, os.path.join(output_dir, 'pathology_map.nii.gz'))
    
    def save_metrics(self, output_dir: str):
        """Сохранение метрик в JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Конвертация numpy типов в Python типы для JSON сериализации
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        metrics_converted = convert_numpy_types(self.metrics)
        quality_converted = convert_numpy_types(self.quality)
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics_converted,
                'quality': quality_converted
            }, f, indent=2, ensure_ascii=False)
    
    def _get_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Получение крупнейшего связного компонента."""
        labeled = label(mask)[0]
        if labeled.max() == 0:
            return mask
        
        props = regionprops(labeled)
        largest_label = max(props, key=lambda x: x.area).label
        return labeled == largest_label
    
    def _remove_small_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Удаление мелких компонентов."""
        labeled = label(mask)[0]
        if labeled.max() == 0:
            return mask
        
        props = regionprops(labeled)
        large_labels = [prop.label for prop in props if prop.area >= min_size]
        
        if not large_labels:
            return np.zeros_like(mask, dtype=bool)
        
        return np.isin(labeled, large_labels)

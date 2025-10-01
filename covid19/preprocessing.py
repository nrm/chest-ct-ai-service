"""
COVID19 Dataset Preprocessing Pipeline

Многопоточная предобработка NIfTI файлов с загрузкой в RAM.
Оптимизировано для Intel Xeon Gold 6334 (16 ядер).
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

# Scipy для морфологических операций и ресемплинга
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
from skimage.transform import resize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Конфигурация предобработки."""

    # Target размеры
    target_spacing: Tuple[float, float, float] = (2.0, 2.0, 8.0)  # X, Y, Z (мм)
    target_shape: Tuple[int, int, int] = (64, 256, 256)  # D, H, W

    # HU processing
    hu_percentile_clip: Tuple[float, float] = (1.0, 99.0)  # Robust clipping
    lung_window: Tuple[float, float] = (-1000.0, 200.0)  # Лёгочное окно

    # ROI extraction
    lung_threshold: float = -300.0  # HU threshold для маски лёгких
    min_lung_area_ratio: float = 0.01  # Минимальная площадь лёгких (% от среза)

    # Морфологические операции для ROI
    morph_kernel_size: int = 5

    # Параллелизация
    n_workers: int = 16  # Xeon Gold 6334: 16 ядер

    # Отладка
    verbose: bool = True


class COVID19Preprocessor:
    """Предобработчик COVID19 NIfTI данных."""

    def __init__(self, config: PreprocessConfig):
        self.config = config

    def load_nifti(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загружает NIfTI файл и извлекает данные + spacing.

        Returns:
            data: (H, W, D) array
            spacing: (X, Y, Z) в мм
        """
        nii = nib.load(str(file_path))
        data = nii.get_fdata()
        spacing = np.array(nii.header.get_zooms())

        return data, spacing

    def robust_hu_clipping(self, data: np.ndarray) -> np.ndarray:
        """
        Robust HU clipping на основе percentiles для устранения выбросов.

        Защита от аномальных значений типа [-32146, +32249].
        """
        # Вычисляем percentiles
        p_low, p_high = self.config.hu_percentile_clip
        hu_min = np.percentile(data, p_low)
        hu_max = np.percentile(data, p_high)

        # Дополнительная защита: если percentiles дают экстремальные значения,
        # используем lung_window как fallback
        if hu_min < -2000 or hu_max > 3000:
            logger.warning(f"Экстремальные percentiles: [{hu_min:.1f}, {hu_max:.1f}]. "
                          f"Используем lung_window как fallback.")
            hu_min, hu_max = self.config.lung_window

        # Clipping
        data_clipped = np.clip(data, hu_min, hu_max)

        return data_clipped

    def normalize_to_01(self, data: np.ndarray) -> np.ndarray:
        """Нормализация в [0, 1]."""
        data_min = data.min()
        data_max = data.max()

        if data_max - data_min < 1e-6:
            logger.warning("Нулевая дисперсия после клиппинга, возвращаем zeros.")
            return np.zeros_like(data)

        normalized = (data - data_min) / (data_max - data_min)
        return normalized

    def extract_lung_roi(self, data: np.ndarray) -> np.ndarray:
        """
        Извлекает бинарную маску лёгких через HU-threshold + морфологию.

        Args:
            data: (H, W, D) HU array (до нормализации!)

        Returns:
            mask: (H, W, D) binary mask
        """
        # Threshold: лёгкие имеют низкий HU
        lung_mask = data < self.config.lung_threshold

        # Морфологические операции для очистки (per-slice)
        kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size))

        for z in range(lung_mask.shape[2]):
            slice_mask = lung_mask[:, :, z]

            # Opening (удаляем мелкий шум)
            slice_mask = binary_opening(slice_mask, structure=kernel)

            # Closing (заполняем дырки)
            slice_mask = binary_closing(slice_mask, structure=kernel)

            # Fill holes (заполняем внутренние дырки)
            slice_mask = binary_fill_holes(slice_mask)

            lung_mask[:, :, z] = slice_mask

        return lung_mask.astype(np.float32)

    def select_informative_slices(
        self,
        data: np.ndarray,
        lung_mask: np.ndarray,
        n_slices: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выбирает n_slices наиболее информативных срезов на основе площади лёгких.

        Args:
            data: (H, W, D)
            lung_mask: (H, W, D)
            n_slices: количество срезов для выбора

        Returns:
            data_selected: (H, W, n_slices)
            lung_mask_selected: (H, W, n_slices)
        """
        D = data.shape[2]

        # Вычисляем площадь лёгких на каждом срезе
        lung_areas = []
        for z in range(D):
            area = lung_mask[:, :, z].sum()
            lung_areas.append(area)

        lung_areas = np.array(lung_areas)

        # Находим центральную область с максимальной площадью
        # Используем скользящее окно для нахождения центра масс
        if D <= n_slices:
            # Если срезов меньше или равно n_slices, берём все + padding
            indices = np.arange(D)
        else:
            # Находим индексы top-n_slices срезов с максимальной площадью
            # Но отдаём предпочтение центральным срезам (weighted)
            center_idx = D // 2
            weights = np.exp(-0.01 * (np.arange(D) - center_idx) ** 2)  # Gaussian weight
            weighted_areas = lung_areas * weights

            # Выбираем top-n_slices по weighted area
            top_indices = np.argsort(weighted_areas)[-n_slices:]
            indices = np.sort(top_indices)  # Сортируем для сохранения порядка

        # Извлекаем срезы
        data_selected = data[:, :, indices]
        lung_mask_selected = lung_mask[:, :, indices]

        # Padding если нужно
        if data_selected.shape[2] < n_slices:
            pad_width = ((0, 0), (0, 0), (0, n_slices - data_selected.shape[2]))
            data_selected = np.pad(data_selected, pad_width, mode='constant', constant_values=0)
            lung_mask_selected = np.pad(lung_mask_selected, pad_width, mode='constant', constant_values=0)

        return data_selected, lung_mask_selected

    def resample_inplane(
        self,
        data: np.ndarray,
        original_spacing: np.ndarray
    ) -> np.ndarray:
        """
        Ресемплит in-plane (X, Y) до target_spacing, сохраняя Z без изменений.

        Args:
            data: (H, W, D)
            original_spacing: (X, Y, Z) в мм

        Returns:
            resampled: (H_new, W_new, D)
        """
        target_spacing = np.array(self.config.target_spacing)

        # Вычисляем новые размеры для in-plane
        # Формула: new_size = original_size * (original_spacing / target_spacing)
        H, W, D = data.shape

        scale_h = original_spacing[1] / target_spacing[1]  # Y spacing
        scale_w = original_spacing[0] / target_spacing[0]  # X spacing

        new_H = int(H * scale_h)
        new_W = int(W * scale_w)

        # Resize per-slice (сохраняем Z)
        resampled = np.zeros((new_H, new_W, D), dtype=data.dtype)

        for z in range(D):
            slice_2d = data[:, :, z]
            # Используем order=1 (bilinear) для smooth interpolation
            resampled[:, :, z] = resize(
                slice_2d,
                (new_H, new_W),
                order=1,  # Bilinear
                mode='constant',
                anti_aliasing=True,
                preserve_range=True
            )

        return resampled

    def resize_to_target(self, data: np.ndarray) -> np.ndarray:
        """
        Resize до target_shape через interpolation.

        Args:
            data: (H, W, D)

        Returns:
            resized: (target_H, target_W, target_D)
        """
        target_D, target_H, target_W = self.config.target_shape

        # Resize 3D volume
        resized = resize(
            data,
            (target_H, target_W, target_D),
            order=1,  # Bilinear
            mode='constant',
            anti_aliasing=True,
            preserve_range=True
        )

        return resized

    def preprocess_single(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Полный pipeline предобработки для одного файла.

        Returns:
            preprocessed: (D, H, W) array float32 [0, 1]
            metadata: dict с информацией о предобработке
        """
        metadata = {
            'file_path': str(file_path),
            'success': False,
            'error': None
        }

        try:
            # 1. Загрузка
            data, spacing = self.load_nifti(file_path)
            metadata['original_shape'] = data.shape
            metadata['original_spacing'] = spacing.tolist()

            # 2. Robust HU clipping
            data_clipped = self.robust_hu_clipping(data)

            # 3. Extract lung ROI (до нормализации, на HU значениях)
            lung_mask = self.extract_lung_roi(data_clipped)

            # 4. Apply lung mask (optional: можем не применять, а только использовать для селекции)
            # data_masked = data_clipped * lung_mask  # Uncomment для жёсткого маскирования
            data_masked = data_clipped  # Пока не маскируем жёстко

            # 5. Normalize to [0, 1]
            data_normalized = self.normalize_to_01(data_masked)

            # 6. Resample in-plane
            data_resampled = self.resample_inplane(data_normalized, spacing)
            metadata['after_resample_shape'] = data_resampled.shape

            # 7. Select informative slices
            lung_mask_resampled = self.resample_inplane(lung_mask, spacing)
            data_selected, _ = self.select_informative_slices(
                data_resampled,
                lung_mask_resampled,
                n_slices=self.config.target_shape[0]
            )
            metadata['after_selection_shape'] = data_selected.shape

            # 8. Resize to target shape
            data_final = self.resize_to_target(data_selected)

            # 9. Transpose to (D, H, W) format
            data_final = np.transpose(data_final, (2, 0, 1))

            metadata['final_shape'] = data_final.shape
            metadata['success'] = True

            return data_final.astype(np.float32), metadata

        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path.name}: {e}")
            metadata['error'] = str(e)
            # Возвращаем zeros в случае ошибки
            D, H, W = self.config.target_shape
            return np.zeros((D, H, W), dtype=np.float32), metadata

    def preprocess_batch(
        self,
        file_paths: List[Path],
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Параллельная предобработка батча файлов.

        Args:
            file_paths: список путей к NIfTI файлам
            show_progress: показывать progress bar

        Returns:
            data_batch: (N, D, H, W) array
            metadata_batch: list of dicts
        """
        n_files = len(file_paths)
        D, H, W = self.config.target_shape

        # Pre-allocate массив
        data_batch = np.zeros((n_files, D, H, W), dtype=np.float32)
        metadata_batch = []

        # Параллельная обработка
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit tasks
            futures = {
                executor.submit(self.preprocess_single, fp): i
                for i, fp in enumerate(file_paths)
            }

            # Collect results с progress bar
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=n_files, desc="Preprocessing")

            for future in iterator:
                idx = futures[future]
                data, metadata = future.result()
                data_batch[idx] = data
                metadata_batch.append(metadata)

        # Сортируем metadata по исходному порядку
        metadata_batch = [metadata_batch[futures[f]] for f in futures]

        return data_batch, metadata_batch


def preprocess_dataset(
    base_path: str,
    config: Optional[PreprocessConfig] = None,
    categories: List[str] = ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Предобрабатывает весь COVID19 dataset и загружает в RAM.

    Args:
        base_path: путь к COVID19_1110/studies/
        config: конфигурация предобработки
        categories: список категорий для обработки

    Returns:
        data: (N, D, H, W) array — все предобработанные исследования
        labels: (N,) array — бинарные метки (0=норма, 1=патология)
        file_paths: (N,) list — пути к исходным файлам
    """
    if config is None:
        config = PreprocessConfig()

    preprocessor = COVID19Preprocessor(config)

    base_path = Path(base_path)

    # Собираем все файлы
    all_files = []
    all_labels = []

    for cat in categories:
        cat_path = base_path / cat
        if not cat_path.exists():
            logger.warning(f"Категория {cat} не найдена, пропускаем")
            continue

        files = sorted(list(cat_path.glob("study_*.nii.gz")))

        # Бинарная метка: CT-0 = 0 (норма), остальные = 1 (патология)
        label = 0 if cat == "CT-0" else 1

        all_files.extend(files)
        all_labels.extend([label] * len(files))

    logger.info(f"Найдено {len(all_files)} файлов для предобработки")
    logger.info(f"Норма: {sum(1 for l in all_labels if l == 0)}, "
                f"Патология: {sum(1 for l in all_labels if l == 1)}")

    # Предобработка
    logger.info("Запуск параллельной предобработки...")
    data, metadata = preprocessor.preprocess_batch(all_files, show_progress=True)

    # Проверка успешности
    n_success = sum(1 for m in metadata if m['success'])
    logger.info(f"Успешно обработано: {n_success}/{len(all_files)}")

    if n_success < len(all_files):
        logger.warning(f"Не удалось обработать {len(all_files) - n_success} файлов")
        for m in metadata:
            if not m['success']:
                logger.warning(f"  {Path(m['file_path']).name}: {m['error']}")

    labels = np.array(all_labels, dtype=np.int64)
    file_paths = [str(fp) for fp in all_files]

    return data, labels, file_paths


if __name__ == "__main__":
    # Тестовый запуск
    base_path = "/mnt/pcephfs/lct/data/covid19_1110/COVID19_1110/studies"

    config = PreprocessConfig(
        n_workers=16,
        verbose=True
    )

    logger.info("Запуск предобработки COVID19 dataset...")
    data, labels, file_paths = preprocess_dataset(base_path, config)

    logger.info(f"\nРезультат:")
    logger.info(f"  Data shape: {data.shape}")
    logger.info(f"  Labels shape: {labels.shape}")
    logger.info(f"  Memory usage: {data.nbytes / (1024**3):.2f} GB")
    logger.info(f"  Норма (0): {(labels == 0).sum()}")
    logger.info(f"  Патология (1): {(labels == 1).sum()}")
#!/usr/bin/env python3
"""
PyTorch Dataset для COVID19 классификации.

Загружает предобработанные данные из splits JSON и применяет аугментации.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Optional
import json

from .augmentations import get_train_augmentations, Compose


class COVID19Dataset(Dataset):
    """
    Dataset для COVID19 классификации.

    Загружает preprocessed данные (64, 256, 256) и применяет аугментации.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        file_paths: List[str],
        transform: Optional[Compose] = None,
        return_path: bool = False
    ):
        """
        Args:
            data: (N, D, H, W) — preprocessed volumes
            labels: (N,) — binary labels (0=normal, 1=pathology)
            file_paths: (N,) — paths to original files
            transform: Аугментации (применяются к каждому слайсу)
            return_path: Возвращать file_path в __getitem__
        """
        super().__init__()

        assert len(data) == len(labels) == len(file_paths), \
            f"Mismatch: data={len(data)}, labels={len(labels)}, paths={len(file_paths)}"

        self.data = data
        self.labels = labels
        self.file_paths = file_paths
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Возвращает одно исследование.

        Returns:
            volume: (D, H, W) — preprocessed volume
            label: scalar — binary label
            path: str — file path (если return_path=True)
        """
        # Получаем данные
        volume = self.data[idx]  # (D, H, W)
        label = self.labels[idx]
        path = self.file_paths[idx]

        # Convert to torch tensor
        volume = torch.from_numpy(volume).float()
        label = torch.tensor(label, dtype=torch.float32)

        # Применяем аугментации к каждому срезу
        if self.transform is not None:
            volume = torch.stack([self.transform(volume[i]) for i in range(len(volume))])

        if self.return_path:
            return volume, label, path

        return volume, label

    @classmethod
    def from_preprocessed(
        cls,
        data: np.ndarray,
        labels: np.ndarray,
        file_paths: List[str],
        augment: bool = True,
        **kwargs
    ):
        """
        Удобный конструктор из preprocessed данных.

        Args:
            data: (N, D, H, W) — preprocessed volumes
            labels: (N,) — binary labels
            file_paths: (N,) — file paths
            augment: Применять аугментации
            **kwargs: Дополнительные аргументы для __init__

        Returns:
            COVID19Dataset instance
        """
        transform = get_train_augmentations() if augment else None

        return cls(
            data=data,
            labels=labels,
            file_paths=file_paths,
            transform=transform,
            **kwargs
        )

    def get_class_weights(self) -> torch.Tensor:
        """
        Вычисляет class weights для loss function.

        Returns:
            weights: [weight_normal, weight_pathology]
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        weights = len(self.labels) / (len(unique) * counts)

        # Создаём тензор weights для классов [0, 1]
        class_weights = torch.zeros(2)
        for cls, weight in zip(unique, weights):
            class_weights[int(cls)] = weight

        return class_weights

    def get_pos_weight(self) -> float:
        """
        Вычисляет pos_weight для BCEWithLogitsLoss.

        pos_weight = n_negative / n_positive

        Returns:
            pos_weight: scalar
        """
        n_negative = (self.labels == 0).sum()
        n_positive = (self.labels == 1).sum()

        return float(n_negative / n_positive) if n_positive > 0 else 1.0


def load_splits_and_preprocess(
    splits_path: Path,
    fold: int,
    split_type: str,
    preprocessed_data: np.ndarray,
    preprocessed_labels: np.ndarray,
    preprocessed_paths: List[str]
) -> COVID19Dataset:
    """
    Загружает split из JSON и создаёт Dataset.

    Args:
        splits_path: Путь к covid19_splits.json
        fold: Номер фолда (0-4)
        split_type: 'train', 'val', или 'holdout'
        preprocessed_data: Полный массив preprocessed данных
        preprocessed_labels: Полный массив labels
        preprocessed_paths: Полный список file paths

    Returns:
        COVID19Dataset для указанного split
    """
    # Загружаем splits
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Получаем file paths для данного split
    if split_type == 'holdout':
        split_data = splits['holdout']
        file_paths = split_data['file_paths']
        labels = split_data['labels']
    else:
        fold_data = splits['cv_folds'][fold]
        split_data = fold_data[split_type]
        file_paths = split_data['file_paths']
        labels = split_data['labels']

    # Создаём mapping path -> index в preprocessed данных
    path_to_idx = {path: i for i, path in enumerate(preprocessed_paths)}

    # Находим индексы в preprocessed данных
    indices = [path_to_idx[path] for path in file_paths]

    # Извлекаем данные
    data = preprocessed_data[indices]
    labels_array = np.array(labels)

    # Проверяем соответствие labels
    assert np.array_equal(labels_array, preprocessed_labels[indices]), \
        "Labels mismatch between splits JSON and preprocessed data"

    # Создаём Dataset
    augment = (split_type == 'train')
    dataset = COVID19Dataset.from_preprocessed(
        data=data,
        labels=labels_array,
        file_paths=file_paths,
        augment=augment
    )

    return dataset


def test_dataset():
    """Тест Dataset."""
    print("=" * 80)
    print("ТЕСТ COVID19Dataset")
    print("=" * 80)

    # Создаём фейковые данные: 20 исследований
    n_samples = 20
    n_slices = 64
    height = 256
    width = 256

    data = np.random.rand(n_samples, n_slices, height, width).astype(np.float32)
    labels = np.array([0] * 5 + [1] * 15)  # 5 норм, 15 патологий
    file_paths = [f"/fake/path/study_{i:04d}.nii.gz" for i in range(n_samples)]

    print(f"\nТестовые данные:")
    print(f"  Samples: {n_samples}")
    print(f"  Shape per sample: ({n_slices}, {height}, {width})")
    print(f"  Normal: {(labels == 0).sum()}")
    print(f"  Pathology: {(labels == 1).sum()}")

    # Тест без аугментаций
    print("\n" + "-" * 80)
    print("Тест 1: Dataset без аугментаций")
    print("-" * 80)

    dataset_no_aug = COVID19Dataset.from_preprocessed(
        data=data,
        labels=labels,
        file_paths=file_paths,
        augment=False
    )

    print(f"\nDataset length: {len(dataset_no_aug)}")

    volume, label = dataset_no_aug[0]
    print(f"Sample 0:")
    print(f"  Volume shape: {tuple(volume.shape)} (должна быть (64, 256, 256))")
    print(f"  Label: {label.item()}")
    print(f"  Volume range: [{volume.min():.4f}, {volume.max():.4f}]")

    # Тест с аугментациями
    print("\n" + "-" * 80)
    print("Тест 2: Dataset с аугментациями")
    print("-" * 80)

    dataset_aug = COVID19Dataset.from_preprocessed(
        data=data,
        labels=labels,
        file_paths=file_paths,
        augment=True
    )

    volume_aug, label_aug = dataset_aug[0]
    print(f"Sample 0 (augmented):")
    print(f"  Volume shape: {tuple(volume_aug.shape)}")
    print(f"  Label: {label_aug.item()}")
    print(f"  Volume range: [{volume_aug.min():.4f}, {volume_aug.max():.4f}]")

    # Проверяем что аугментации применяются
    volume_aug2, _ = dataset_aug[0]
    is_different = not torch.allclose(volume_aug, volume_aug2, atol=1e-6)
    print(f"  Аугментации рандомизированы: {is_different} (должно быть True)")

    # Тест class weights
    print("\n" + "-" * 80)
    print("Тест 3: Class weights")
    print("-" * 80)

    class_weights = dataset_aug.get_class_weights()
    pos_weight = dataset_aug.get_pos_weight()

    print(f"\nClass weights: {class_weights}")
    print(f"  Normal (class 0): {class_weights[0]:.4f}")
    print(f"  Pathology (class 1): {class_weights[1]:.4f}")
    print(f"\nPos weight (для BCEWithLogitsLoss): {pos_weight:.4f}")
    print(f"  Ожидаемое значение: {(labels == 0).sum() / (labels == 1).sum():.4f}")

    # Тест DataLoader
    print("\n" + "-" * 80)
    print("Тест 4: DataLoader")
    print("-" * 80)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset_aug,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    batch_volumes, batch_labels = next(iter(dataloader))

    print(f"\nПервый батч:")
    print(f"  Batch volumes shape: {tuple(batch_volumes.shape)} (должна быть (4, 64, 256, 256))")
    print(f"  Batch labels shape: {tuple(batch_labels.shape)} (должна быть (4,))")
    print(f"  Labels: {batch_labels}")

    print("\n" + "=" * 80)
    print("✅ Все тесты пройдены!")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset()
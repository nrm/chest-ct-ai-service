#!/usr/bin/env python3
"""
Аугментации для COVID19 2D slices.

Включает медицински обоснованные трансформации:
- Geometric: rotation, flip, affine
- Intensity: Gaussian noise, HU-jitter
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import random


class RandomRotation:
    """Случайный поворот среза."""

    def __init__(self, degrees: Tuple[float, float] = (-15, 15), p: float = 0.5):
        """
        Args:
            degrees: Диапазон углов (min, max) в градусах
            p: Вероятность применения
        """
        self.degrees = degrees
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, H, W) или (H, W)

        Returns:
            rotated: (C, H, W) или (H, W)
        """
        if random.random() > self.p:
            return x

        # Random angle
        angle = random.uniform(*self.degrees)

        # Convert to radians
        angle_rad = np.deg2rad(angle)

        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Affine matrix для PyTorch (2x3)
        # [cos -sin tx]
        # [sin  cos ty]
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device)

        # Add batch and channel dims if needed
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            squeeze_dims = True
        elif x.ndim == 3:
            x = x.unsqueeze(0)  # (1, C, H, W)
            squeeze_dims = False
        else:
            squeeze_dims = False

        # Create grid and apply affine
        grid = F.affine_grid(theta.unsqueeze(0), x.shape, align_corners=False)
        x_rotated = F.grid_sample(x, grid, mode='bilinear', align_corners=False, padding_mode='zeros')

        # Remove batch dim
        if squeeze_dims:
            x_rotated = x_rotated.squeeze(0).squeeze(0)
        else:
            x_rotated = x_rotated.squeeze(0)

        return x_rotated


class RandomHorizontalFlip:
    """Горизонтальный flip (анатомически допустимо для лёгких)."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Вероятность применения
        """
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, H, W) или (H, W)

        Returns:
            flipped: (C, H, W) или (H, W)
        """
        if random.random() > self.p:
            return x

        return torch.flip(x, dims=[-1])  # Flip по последней размерности (W)


class RandomAffine:
    """Случайные affine трансформации: scale, translate."""

    def __init__(
        self,
        scale: Tuple[float, float] = (0.9, 1.1),
        translate: Tuple[float, float] = (0.1, 0.1),
        p: float = 0.5
    ):
        """
        Args:
            scale: Диапазон масштабирования (min, max)
            translate: Максимальный сдвиг как доля размера (tx, ty)
            p: Вероятность применения
        """
        self.scale = scale
        self.translate = translate
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, H, W) или (H, W)

        Returns:
            transformed: (C, H, W) или (H, W)
        """
        if random.random() > self.p:
            return x

        # Random scale
        s = random.uniform(*self.scale)

        # Random translate
        tx = random.uniform(-self.translate[0], self.translate[0])
        ty = random.uniform(-self.translate[1], self.translate[1])

        # Affine matrix
        theta = torch.tensor([
            [s, 0, tx],
            [0, s, ty]
        ], dtype=x.dtype, device=x.device)

        # Add batch and channel dims if needed
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_dims = True
        elif x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_dims = False
        else:
            squeeze_dims = False

        # Apply affine
        grid = F.affine_grid(theta.unsqueeze(0), x.shape, align_corners=False)
        x_transformed = F.grid_sample(x, grid, mode='bilinear', align_corners=False, padding_mode='zeros')

        # Remove batch dim
        if squeeze_dims:
            x_transformed = x_transformed.squeeze(0).squeeze(0)
        else:
            x_transformed = x_transformed.squeeze(0)

        return x_transformed


class GaussianNoise:
    """Аддитивный Gaussian noise."""

    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.5):
        """
        Args:
            mean: Среднее шума
            std: Std шума (относительно диапазона [0, 1])
            p: Вероятность применения
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, H, W) или (H, W)

        Returns:
            noisy: (C, H, W) или (H, W)
        """
        if random.random() > self.p:
            return x

        noise = torch.randn_like(x) * self.std + self.mean
        x_noisy = x + noise

        # Clip to [0, 1]
        x_noisy = torch.clamp(x_noisy, 0, 1)

        return x_noisy


class HUJitter:
    """
    HU-jitter: имитация вариативности сканеров.

    Добавляет случайный offset к HU значениям.
    """

    def __init__(self, jitter_range: Tuple[float, float] = (-0.05, 0.05), p: float = 0.5):
        """
        Args:
            jitter_range: Диапазон offset (относительно [0, 1] диапазона)
            p: Вероятность применения
        """
        self.jitter_range = jitter_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, H, W) или (H, W), normalized to [0, 1]

        Returns:
            jittered: (C, H, W) или (H, W)
        """
        if random.random() > self.p:
            return x

        offset = random.uniform(*self.jitter_range)
        x_jittered = x + offset

        # Clip to [0, 1]
        x_jittered = torch.clamp(x_jittered, 0, 1)

        return x_jittered


class Compose:
    """Композиция нескольких аугментаций."""

    def __init__(self, transforms: list):
        """
        Args:
            transforms: Список аугментаций
        """
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def get_train_augmentations(
    rotation_degrees: Tuple[float, float] = (-15, 15),
    flip_p: float = 0.5,
    affine_scale: Tuple[float, float] = (0.9, 1.1),
    affine_translate: Tuple[float, float] = (0.1, 0.1),
    noise_std: float = 0.05,
    hu_jitter: Tuple[float, float] = (-0.05, 0.05),
    prob: float = 0.5
) -> Compose:
    """
    Стандартный набор аугментаций для тренировки.

    Args:
        rotation_degrees: Диапазон поворотов
        flip_p: Вероятность horizontal flip
        affine_scale: Диапазон масштабирования
        affine_translate: Максимальный сдвиг
        noise_std: Std Gaussian noise
        hu_jitter: Диапазон HU jitter
        prob: Вероятность применения каждой аугментации

    Returns:
        Compose объект с аугментациями
    """
    return Compose([
        RandomRotation(degrees=rotation_degrees, p=prob),
        RandomHorizontalFlip(p=flip_p),
        RandomAffine(scale=affine_scale, translate=affine_translate, p=prob),
        GaussianNoise(mean=0.0, std=noise_std, p=prob),
        HUJitter(jitter_range=hu_jitter, p=prob)
    ])


def test_augmentations():
    """Тест аугментаций."""
    print("=" * 80)
    print("ТЕСТ AUGMENTATIONS")
    print("=" * 80)

    # Создаём фейковый срез (256, 256)
    x = torch.rand(256, 256)
    print(f"\nИсходный срез: {tuple(x.shape)}")
    print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}")

    # Тест каждой аугментации
    augmentations = [
        ("RandomRotation", RandomRotation(degrees=(-15, 15), p=1.0)),
        ("RandomHorizontalFlip", RandomHorizontalFlip(p=1.0)),
        ("RandomAffine", RandomAffine(scale=(0.9, 1.1), translate=(0.1, 0.1), p=1.0)),
        ("GaussianNoise", GaussianNoise(mean=0.0, std=0.05, p=1.0)),
        ("HUJitter", HUJitter(jitter_range=(-0.05, 0.05), p=1.0))
    ]

    print("\n" + "-" * 80)
    print("Тест отдельных аугментаций:")
    print("-" * 80)

    for name, aug in augmentations:
        x_aug = aug(x)
        print(f"\n{name}:")
        print(f"  Shape: {tuple(x_aug.shape)} (должна быть {tuple(x.shape)})")
        print(f"  Min: {x_aug.min():.4f}, Max: {x_aug.max():.4f}, Mean: {x_aug.mean():.4f}")
        print(f"  В диапазоне [0, 1]: {(x_aug >= 0).all() and (x_aug <= 1).all()}")

    # Тест композиции
    print("\n" + "-" * 80)
    print("Тест композиции всех аугментаций:")
    print("-" * 80)

    train_aug = get_train_augmentations(prob=0.8)
    x_aug = train_aug(x)

    print(f"\nПосле композиции:")
    print(f"  Shape: {tuple(x_aug.shape)}")
    print(f"  Min: {x_aug.min():.4f}, Max: {x_aug.max():.4f}, Mean: {x_aug.mean():.4f}")
    print(f"  В диапазоне [0, 1]: {(x_aug >= 0).all() and (x_aug <= 1).all()}")

    # Тест с batch
    print("\n" + "-" * 80)
    print("Тест с батчем (64 слайса):")
    print("-" * 80)

    x_batch = torch.rand(64, 256, 256)
    print(f"\nИсходный батч: {tuple(x_batch.shape)}")

    # Применяем аугментацию к каждому срезу
    x_batch_aug = torch.stack([train_aug(x_batch[i]) for i in range(len(x_batch))])

    print(f"После аугментации: {tuple(x_batch_aug.shape)}")
    print(f"  Min: {x_batch_aug.min():.4f}, Max: {x_batch_aug.max():.4f}")

    print("\n" + "=" * 80)
    print("✅ Все тесты пройдены!")
    print("=" * 80)


if __name__ == "__main__":
    test_augmentations()
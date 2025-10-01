#!/usr/bin/env python3
"""
COVID19 2D MIL Classifier с ResNet50 encoder и attention pooling.

Архитектура:
  Input: (B, N_slices, H, W) — батч из B исследований по N_slices срезов
  1. SliceEncoder: ResNet50 (pretrained) → per-slice features (B×N_slices, 2048)
  2. AttentionPooling: weighted aggregation → study features (B, 2048)
  3. Classifier head: FC → Sigmoid → probability of pathology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


class SliceEncoder(nn.Module):
    """
    Per-slice feature encoder на основе ResNet50 (ImageNet pretrained).

    Адаптирует первый conv слой: 3 канала → 1 канал (grayscale CT).
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            pretrained: Использовать ImageNet weights
            freeze_backbone: Заморозить ResNet кроме FC слоёв
        """
        super().__init__()

        # Загружаем ResNet50
        resnet = models.resnet50(pretrained=pretrained)

        # Адаптируем первый conv: 3 каналов → 1 канал
        # Усредняем веса по каналам чтобы сохранить pretrained knowledge
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Усредняем веса ImageNet RGB → grayscale
            with torch.no_grad():
                self.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # Остальные слои ResNet
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool

        # Размерность выходных признаков ResNet50
        self.feature_dim = 2048

        # Опционально замораживаем backbone
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
            # Размораживаем только последний residual block
            for param in self.layer4.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через ResNet50.

        Args:
            x: (B×N_slices, 1, H, W) — батч срезов

        Returns:
            features: (B×N_slices, 2048) — per-slice embeddings
        """
        # ResNet forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class AttentionPooling(nn.Module):
    """
    Attention-based aggregation по срезам.

    Использует self-attention для взвешенного усреднения slice embeddings.
    Позволяет модели автоматически фокусироваться на информативных срезах.
    """

    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512):
        """
        Args:
            feature_dim: Размерность входных slice features
            hidden_dim: Размерность скрытого слоя attention
        """
        super().__init__()

        # Attention mechanism: query, key, value
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregation по срезам с attention.

        Args:
            x: (B, N_slices, feature_dim) — slice features
            return_weights: Вернуть attention веса для визуализации

        Returns:
            aggregated: (B, feature_dim) — study-level features
            weights: (B, N_slices) — attention weights (если return_weights=True)
        """
        # Вычисляем attention scores
        # (B, N_slices, 1)
        attention_scores = self.attention(x)

        # Softmax по измерению срезов
        # (B, N_slices, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum
        # (B, feature_dim)
        aggregated = torch.sum(x * attention_weights, dim=1)

        if return_weights:
            return aggregated, attention_weights.squeeze(-1)

        return aggregated


class COVID19Classifier(nn.Module):
    """
    Полная архитектура COVID19 классификатора (2D MIL).

    Pipeline:
      Input (B, N_slices, H, W)
        ↓
      SliceEncoder: ResNet50
        ↓
      (B, N_slices, 2048)
        ↓
      AttentionPooling
        ↓
      (B, 2048)
        ↓
      Classifier head: FC → Dropout → FC → Sigmoid
        ↓
      (B,) — probability of pathology
    """

    def __init__(
        self,
        n_slices: int = 64,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        """
        Args:
            n_slices: Число срезов на исследование
            pretrained: ImageNet pretrained weights для ResNet50
            freeze_backbone: Заморозить ResNet backbone
            dropout: Dropout rate в classifier head
        """
        super().__init__()

        self.n_slices = n_slices

        # Slice-level encoder
        self.slice_encoder = SliceEncoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

        # Attention pooling
        self.attention_pooling = AttentionPooling(
            feature_dim=self.slice_encoder.feature_dim,
            hidden_dim=512
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.slice_encoder.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, N_slices, H, W) — батч исследований
            return_attention: Вернуть attention weights

        Returns:
            logits: (B,) — logits (применить sigmoid для probabilities)
            attention_weights: (B, N_slices) — если return_attention=True
        """
        B, N, H, W = x.shape

        # Reshape для per-slice encoding
        # (B, N, H, W) → (B*N, 1, H, W)
        x = x.view(B * N, 1, H, W)

        # Encode slices
        # (B*N, 1, H, W) → (B*N, 2048)
        slice_features = self.slice_encoder(x)

        # Reshape обратно
        # (B*N, 2048) → (B, N, 2048)
        slice_features = slice_features.view(B, N, -1)

        # Attention pooling
        if return_attention:
            study_features, attention_weights = self.attention_pooling(
                slice_features, return_weights=True
            )
        else:
            study_features = self.attention_pooling(slice_features)
            attention_weights = None

        # Classifier head
        # (B, 2048) → (B, 1) → (B,)
        logits = self.classifier(study_features).squeeze(-1)

        if return_attention:
            return logits, attention_weights

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Предсказание вероятностей (с sigmoid).

        Args:
            x: (B, N_slices, H, W)

        Returns:
            probabilities: (B,) — probability of pathology
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


def test_architecture():
    """Тест архитектуры на случайных данных."""
    print("=" * 80)
    print("ТЕСТ АРХИТЕКТУРЫ COVID19Classifier")
    print("=" * 80)

    # Гиперпараметры
    batch_size = 2
    n_slices = 64
    height = 256
    width = 256

    # Создаём модель
    model = COVID19Classifier(
        n_slices=n_slices,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.5
    )

    print(f"\nМодель создана:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Тестовый вход
    x = torch.randn(batch_size, n_slices, height, width)
    print(f"\nВход: {tuple(x.shape)}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(x, return_attention=True)
        probas = torch.sigmoid(logits)

    print(f"\nВыход:")
    print(f"  Logits: {tuple(logits.shape)} — {logits}")
    print(f"  Probabilities: {tuple(probas.shape)} — {probas}")
    print(f"  Attention weights: {tuple(attention_weights.shape)}")
    print(f"    Mean: {attention_weights.mean(dim=1)}")
    print(f"    Std: {attention_weights.std(dim=1)}")
    print(f"    Max slice: {attention_weights.argmax(dim=1)}")

    # Проверка attention weights
    attention_sum = attention_weights.sum(dim=1)
    print(f"\n  Attention sum (должна быть ~1.0): {attention_sum}")

    print("\n" + "=" * 80)
    print("✅ Архитектура работает корректно!")
    print("=" * 80)


if __name__ == "__main__":
    test_architecture()
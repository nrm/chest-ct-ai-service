#!/usr/bin/env python3
"""
Loss functions для COVID19 классификации.

Включает:
- FocalLoss: Для борьбы с class imbalance и фокусом на трудных примерах
- WeightedBCEWithLogitsLoss: BCE с class weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss для бинарной классификации.

    Focal Loss = -α(1-p_t)^γ * log(p_t)

    где:
      p_t = p если y=1, иначе (1-p)
      α — weight для positive class
      γ (gamma) — focusing parameter

    Преимущества:
    - Фокусируется на трудных примерах (down-weights easy examples)
    - Помогает с class imbalance
    - γ=2 хорошо работает для медицинских данных

    Reference:
      Lin et al. "Focal Loss for Dense Object Detection" (2017)
      https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        pos_weight: float = None
    ):
        """
        Args:
            alpha: Weight для positive class (pathology)
                  Для imbalance 1:3.37 рекомендуется α ≈ 0.25 (1/4)
            gamma: Focusing parameter. γ=2 — стандартное значение
                  γ=0 → обычный BCE
                  γ↑ → больший фокус на hard examples
            reduction: 'mean', 'sum', или 'none'
            pos_weight: Дополнительный weight для positive class
                       Если задан, умножается на α
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            logits: (B,) — predicted logits (до sigmoid)
            targets: (B,) — ground truth labels (0 или 1)

        Returns:
            loss: scalar tensor
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t: prob если target=1, иначе (1-prob)
        # targets: (B,)
        # probs: (B,)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weight
        if self.pos_weight is not None:
            # Дополнительный weight для positive class
            alpha_weight = self.alpha * self.pos_weight * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute base loss: -log(p_t)
        # Используем BCE с numerically stable log
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Focal Loss = focal_weight * alpha_weight * bce_loss
        focal_loss = focal_weight * alpha_weight * bce_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy с class weights для imbalanced data.

    Простая альтернатива Focal Loss.
    Полезно для baseline экспериментов.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            pos_weight: Weight для positive class (pathology)
                       Для imbalance 1:3.37 рекомендуется pos_weight ≈ 3.37
            label_smoothing: Label smoothing factor [0, 1)
                           0.0 — нет smoothing
                           0.1 — targets становятся [0.05, 0.95]
            reduction: 'mean', 'sum', или 'none'
        """
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            logits: (B,) — predicted logits
            targets: (B,) — ground truth labels

        Returns:
            loss: scalar tensor
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)

        # BCE with logits (numerically stable)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )

        return loss


def test_losses():
    """Тест loss functions."""
    print("=" * 80)
    print("ТЕСТ LOSS FUNCTIONS")
    print("=" * 80)

    # Тестовые данные: 8 примеров
    # 2 нормы (easy), 2 патологии (easy), 2 нормы (hard), 2 патологии (hard)
    batch_size = 8

    # Logits: высокие значения = уверенность в патологии
    logits = torch.tensor([
        -5.0, -4.0,  # Easy normals (low logits → low prob)
        5.0, 4.0,    # Easy pathologies (high logits → high prob)
        0.5, -0.5,   # Hard normals (close to 0.5 prob)
        0.5, -0.5    # Hard pathologies (close to 0.5 prob)
    ])

    targets = torch.tensor([
        0.0, 0.0,    # Normals
        1.0, 1.0,    # Pathologies
        0.0, 0.0,    # Hard normals
        1.0, 1.0     # Hard pathologies
    ])

    print("\nТестовые данные:")
    probs = torch.sigmoid(logits)
    for i in range(batch_size):
        difficulty = "easy" if i < 4 else "hard"
        print(f"  [{i}] target={targets[i]:.0f}, logit={logits[i]:6.2f}, "
              f"prob={probs[i]:.4f} ({difficulty})")

    # Тест 1: Focal Loss
    print("\n" + "-" * 80)
    print("ТЕСТ 1: Focal Loss (α=0.25, γ=2.0)")
    print("-" * 80)

    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
    focal_losses = focal_loss_fn(logits, targets)

    print("\nPer-sample losses:")
    for i in range(batch_size):
        difficulty = "easy" if i < 4 else "hard"
        print(f"  [{i}] loss={focal_losses[i]:.4f} ({difficulty})")

    print(f"\nСреднее:")
    print(f"  Easy examples (0-3): {focal_losses[:4].mean():.4f}")
    print(f"  Hard examples (4-7): {focal_losses[4:].mean():.4f}")
    print(f"  Ratio (hard/easy): {(focal_losses[4:].mean() / focal_losses[:4].mean()):.2f}x")

    total_focal = focal_losses.mean()
    print(f"\nTotal Focal Loss: {total_focal:.4f}")

    # Тест 2: Weighted BCE
    print("\n" + "-" * 80)
    print("ТЕСТ 2: Weighted BCE (pos_weight=3.37, label_smoothing=0.1)")
    print("-" * 80)

    bce_loss_fn = WeightedBCEWithLogitsLoss(
        pos_weight=3.37,
        label_smoothing=0.1,
        reduction='none'
    )
    bce_losses = bce_loss_fn(logits, targets)

    print("\nPer-sample losses:")
    for i in range(batch_size):
        difficulty = "easy" if i < 4 else "hard"
        print(f"  [{i}] loss={bce_losses[i]:.4f} ({difficulty})")

    print(f"\nСреднее:")
    print(f"  Easy examples (0-3): {bce_losses[:4].mean():.4f}")
    print(f"  Hard examples (4-7): {bce_losses[4:].mean():.4f}")
    print(f"  Ratio (hard/easy): {(bce_losses[4:].mean() / bce_losses[:4].mean()):.2f}x")

    total_bce = bce_losses.mean()
    print(f"\nTotal BCE Loss: {total_bce:.4f}")

    # Сравнение
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ")
    print("=" * 80)
    print(f"Focal Loss фокусируется на hard examples сильнее:")
    print(f"  Focal hard/easy ratio: {(focal_losses[4:].mean() / focal_losses[:4].mean()):.2f}x")
    print(f"  BCE hard/easy ratio: {(bce_losses[4:].mean() / bce_losses[:4].mean()):.2f}x")

    print("\n✅ Тесты пройдены!")
    print("=" * 80)


if __name__ == "__main__":
    test_losses()
"""
LUNA16 3D Nodule Detector - Memory-efficient 3D CNN for nodule detection
Optimized for NVIDIA A30 GPU with gradient checkpointing and patch-based processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Conv3DBlock(nn.Module):
    """3D convolutional block with BatchNorm and ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Residual3DBlock(nn.Module):
    """3D Residual block for deeper networks"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv1 = Conv3DBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(residual)
        return F.relu(out)

class AttentionGate3D(nn.Module):
    """3D Attention gate for focusing on relevant features"""

    def __init__(self, gate_channels: int, feature_channels: int, inter_channels: int):
        super().__init__()

        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )

        self.W_feature = nn.Sequential(
            nn.Conv3d(feature_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, feature):
        g1 = self.W_gate(gate)
        x1 = self.W_feature(feature)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return feature * psi

class LUNA16NoduleDetector(nn.Module):
    """
    3D CNN for LUNA16 nodule detection
    Memory-efficient architecture with gradient checkpointing
    Input: 64x64x64 patches, Output: Binary classification (nodule/no-nodule)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        feature_maps: List[int] = [32, 64, 128, 256, 512],
        use_attention: bool = True,
        use_checkpoint: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_checkpoint = use_checkpoint

        # Encoder (contracting path)
        self.encoder_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()

        prev_channels = in_channels
        for i, channels in enumerate(feature_maps):
            # Double conv block
            block = nn.Sequential(
                Conv3DBlock(prev_channels, channels, use_checkpoint=use_checkpoint),
                Conv3DBlock(channels, channels, use_checkpoint=use_checkpoint)
            )
            self.encoder_blocks.append(block)

            # Max pooling (except for last layer)
            if i < len(feature_maps) - 1:
                self.pool_blocks.append(nn.MaxPool3d(2, 2))

            prev_channels = channels

        # Attention gates (if enabled)
        if use_attention:
            self.attention_gates = nn.ModuleList()
            for i in range(len(feature_maps) - 2, -1, -1):
                # Gate comes from upconvolved decoder (has feature_maps[i] channels after upconv)
                gate_channels = feature_maps[i]
                # Skip connection has feature_maps[i] channels
                feature_channels = feature_maps[i]
                inter_channels = feature_channels // 2

                self.attention_gates.append(
                    AttentionGate3D(gate_channels, feature_channels, inter_channels)
                )

        # Decoder (expanding path)
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()

        for i in range(len(feature_maps) - 2, -1, -1):
            # Upconvolution
            upconv = nn.ConvTranspose3d(
                feature_maps[i + 1], feature_maps[i],
                kernel_size=2, stride=2, bias=False
            )
            self.upconv_blocks.append(upconv)

            # Double conv block
            in_channels = feature_maps[i] * 2  # Skip connection
            block = nn.Sequential(
                Conv3DBlock(in_channels, feature_maps[i], use_checkpoint=use_checkpoint),
                Conv3DBlock(feature_maps[i], feature_maps[i], use_checkpoint=use_checkpoint)
            )
            self.decoder_blocks.append(block)

        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_maps[0], feature_maps[0] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(feature_maps[0] // 2, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features: bool = False):
        """
        Forward pass

        Args:
            x: Input tensor (B, 1, D, H, W) - batch of 3D patches
            return_features: Whether to return intermediate features

        Returns:
            logits: Classification logits (B, num_classes)
            features: Optional intermediate features
        """
        # Store encoder features for skip connections
        encoder_features = []

        # Encoder path
        current = x
        for i, encoder_block in enumerate(self.encoder_blocks):
            current = encoder_block(current)
            encoder_features.append(current)

            # Apply pooling (except for bottleneck)
            if i < len(self.pool_blocks):
                current = self.pool_blocks[i](current)

        # Decoder path with skip connections
        decoder_features = []
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            # Upconvolution
            current = upconv(current)

            # Get corresponding encoder feature
            skip_idx = len(encoder_features) - 2 - i
            skip_feature = encoder_features[skip_idx]

            # Apply attention gate if enabled
            if self.use_attention and i < len(self.attention_gates):
                skip_feature = self.attention_gates[i](current, skip_feature)

            # Concatenate skip connection
            current = torch.cat([current, skip_feature], dim=1)

            # Double conv
            current = decoder_block(current)
            decoder_features.append(current)

        # Global pooling and classification
        pooled = self.global_pool(current)  # (B, C, 1, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)

        logits = self.classifier(pooled)

        if return_features:
            return logits, {
                'encoder_features': encoder_features,
                'decoder_features': decoder_features,
                'pooled_features': pooled
            }

        return logits

    def predict_proba(self, x) -> torch.Tensor:
        """Get prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'LUNA16NoduleDetector',
            'architecture': '3D U-Net with attention gates',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'use_attention': self.use_attention,
            'use_checkpoint': self.use_checkpoint
        }

class NoduleDetectorLoss(nn.Module):
    """Custom loss for nodule detection with hard negative mining"""

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        hard_negative_ratio: float = 3.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.hard_negative_ratio = hard_negative_ratio

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with focal loss and hard negative mining

        Args:
            logits: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)

        Returns:
            loss: Computed loss value
        """
        # Compute focal loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss

        # Hard negative mining
        if self.hard_negative_ratio > 0:
            # Separate positive and negative samples
            pos_mask = targets == 1
            neg_mask = targets == 0

            num_pos = pos_mask.sum()
            num_neg_keep = int(num_pos * self.hard_negative_ratio)

            if num_neg_keep > 0 and neg_mask.sum() > num_neg_keep:
                # Select hardest negatives
                neg_losses = focal_loss[neg_mask]
                _, neg_indices = torch.topk(neg_losses, num_neg_keep)

                # Create final mask
                final_mask = pos_mask.clone()
                neg_indices_full = torch.where(neg_mask)[0][neg_indices]
                final_mask[neg_indices_full] = True

                return focal_loss[final_mask].mean()

        return focal_loss.mean()

def create_luna16_detector(**kwargs) -> LUNA16NoduleDetector:
    """Factory function to create LUNA16 nodule detector"""
    return LUNA16NoduleDetector(**kwargs)

if __name__ == "__main__":
    # Test model
    model = LUNA16NoduleDetector()

    # Test input: batch_size=2, channels=1, depth=64, height=64, width=64
    test_input = torch.randn(2, 1, 64, 64, 64)

    # Forward pass
    logits, features = model(test_input, return_features=True)

    print(f"Model info: {model.get_model_info()}")
    print(f"Input shape: {test_input.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Feature shapes:")
    for key, feature_list in features.items():
        if isinstance(feature_list, list):
            print(f"  {key}: {[f.shape for f in feature_list]}")
        else:
            print(f"  {key}: {feature_list.shape}")

    # Test probabilities
    probs = model.predict_proba(test_input)
    print(f"Probabilities: {probs.shape}, Sample: {probs[0]}")

    # Test loss
    targets = torch.tensor([0, 1])
    loss_fn = NoduleDetectorLoss()
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item()}")
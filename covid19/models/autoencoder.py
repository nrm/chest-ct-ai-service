"""
COVID19 Autoencoder for Anomaly Detection
Trains on normal (CT-0) samples to learn reconstruction of healthy chest CTs
Uses reconstruction error as anomaly score for pathology detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SliceEncoder(nn.Module):
    """2D CNN encoder for single CT slice"""

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder: (1, 256, 256) → (latent_dim,)
        self.encoder = nn.Sequential(
            # Block 1: 256x256 → 128x128
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 128x128 → 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 64x64 → 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: 32x32 → 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5: 16x16 → 8x8
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Latent projection: 512*8*8 → latent_dim
        self.fc_encode = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) single slice
        Returns:
            z: (B, latent_dim) latent vector
        """
        features = self.encoder(x)  # (B, 512, 8, 8)
        features = features.view(features.size(0), -1)  # (B, 512*8*8)
        z = self.fc_encode(features)  # (B, latent_dim)
        return z


class SliceDecoder(nn.Module):
    """2D CNN decoder for single CT slice"""

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.latent_dim = latent_dim

        # Latent unprojection: latent_dim → 512*8*8
        self.fc_decode = nn.Linear(latent_dim, 512 * 8 * 8)

        # Decoder: (512, 8, 8) → (1, 256, 256)
        self.decoder = nn.Sequential(
            # Block 1: 8x8 → 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 2: 16x16 → 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 3: 32x32 → 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: 64x64 → 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 5: 128x128 → 256x256
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent vector
        Returns:
            x_recon: (B, 1, H, W) reconstructed slice
        """
        features = self.fc_decode(z)  # (B, 512*8*8)
        features = features.view(-1, 512, 8, 8)  # (B, 512, 8, 8)
        x_recon = self.decoder(features)  # (B, 1, 256, 256)
        return x_recon


class COVID19Autoencoder(nn.Module):
    """
    2.5D Autoencoder for COVID19 chest CT anomaly detection

    Architecture:
    - Processes N slices independently through 2D encoder
    - Aggregates slice features (mean pooling)
    - Reconstructs all slices from aggregated latent

    Training:
    - Only on normal (CT-0) samples
    - Learns to reconstruct healthy chest CTs

    Inference:
    - Reconstruction error → anomaly score
    - High error → likely pathology
    """

    def __init__(
        self,
        num_slices: int = 64,
        latent_dim: int = 256,
        aggregation: str = "mean"  # "mean", "max", or "attention"
    ):
        super().__init__()

        self.num_slices = num_slices
        self.latent_dim = latent_dim
        self.aggregation = aggregation

        # Encoder/decoder for single slices
        self.slice_encoder = SliceEncoder(latent_dim)
        self.slice_decoder = SliceDecoder(latent_dim)

        # Optional: attention mechanism for aggregation
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 4),
                nn.ReLU(),
                nn.Linear(latent_dim // 4, 1)
            )
        else:
            self.attention = None

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode volume to latent representation

        Args:
            x: (B, N, H, W) where N=num_slices

        Returns:
            z_agg: (B, latent_dim) aggregated latent
            z_slices: (B, N, latent_dim) per-slice latents (for debugging)
        """
        B, N, H, W = x.shape

        # Reshape to (B*N, 1, H, W) for batch processing
        x_slices = x.view(B * N, 1, H, W)

        # Encode all slices
        z_slices = self.slice_encoder(x_slices)  # (B*N, latent_dim)
        z_slices = z_slices.view(B, N, self.latent_dim)  # (B, N, latent_dim)

        # Aggregate slice features
        if self.aggregation == "mean":
            z_agg = z_slices.mean(dim=1)  # (B, latent_dim)
        elif self.aggregation == "max":
            z_agg = z_slices.max(dim=1)[0]  # (B, latent_dim)
        elif self.aggregation == "attention":
            # Attention weights: (B, N, 1)
            attn_scores = self.attention(z_slices)  # (B, N, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (B, N, 1)
            z_agg = (z_slices * attn_weights).sum(dim=1)  # (B, latent_dim)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return z_agg, z_slices

    def decode(self, z_agg: torch.Tensor, num_slices: Optional[int] = None) -> torch.Tensor:
        """
        Decode latent to volume reconstruction

        Args:
            z_agg: (B, latent_dim) aggregated latent
            num_slices: number of slices to reconstruct (default: self.num_slices)

        Returns:
            x_recon: (B, N, H, W) reconstructed volume
        """
        if num_slices is None:
            num_slices = self.num_slices

        B = z_agg.size(0)

        # Expand latent for all slices: (B, latent_dim) → (B*N, latent_dim)
        z_expanded = z_agg.unsqueeze(1).expand(B, num_slices, self.latent_dim)
        z_expanded = z_expanded.contiguous().view(B * num_slices, self.latent_dim)

        # Decode all slices
        x_recon_slices = self.slice_decoder(z_expanded)  # (B*N, 1, H, W)

        # Reshape to (B, N, H, W)
        _, _, H, W = x_recon_slices.shape
        x_recon = x_recon_slices.view(B, num_slices, H, W)

        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder forward pass

        Args:
            x: (B, N, H, W) input volume

        Returns:
            x_recon: (B, N, H, W) reconstructed volume
            z_agg: (B, latent_dim) aggregated latent (for regularization)
        """
        z_agg, _ = self.encode(x)
        x_recon = self.decode(z_agg, num_slices=x.size(1))
        return x_recon, z_agg

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute reconstruction error (anomaly score)

        Args:
            x: (B, N, H, W) original volume
            x_recon: (B, N, H, W) reconstructed volume
            reduction: "mean", "sum", or "none"

        Returns:
            error: scalar (mean/sum) or (B,) per-sample errors
        """
        # MSE per pixel: (B, N, H, W)
        mse = (x - x_recon) ** 2

        if reduction == "none":
            # Per-sample error: (B,)
            return mse.view(mse.size(0), -1).mean(dim=1)
        elif reduction == "mean":
            return mse.mean()
        elif reduction == "sum":
            return mse.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score for inference

        Args:
            x: (B, N, H, W) input volume

        Returns:
            scores: (B,) anomaly scores (higher = more anomalous)
        """
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            scores = self.compute_reconstruction_error(x, x_recon, reduction="none")
        return scores


class COVID19AutoencoderLoss(nn.Module):
    """
    Combined loss for autoencoder training

    Components:
    1. Reconstruction loss (MSE)
    2. Optional: latent regularization (KL divergence for VAE-style)
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.0,
        perceptual_weight: float = 0.0
    ):
        super().__init__()

        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z_agg: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss

        Args:
            x: (B, N, H, W) original volume
            x_recon: (B, N, H, W) reconstructed volume
            z_agg: (B, latent_dim) aggregated latent (optional)

        Returns:
            loss: scalar total loss
            loss_dict: dict of individual loss components
        """
        # 1. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        total_loss = self.recon_weight * recon_loss
        loss_dict = {"recon_loss": recon_loss.item()}

        # 2. KL divergence (if using VAE-style latent)
        if self.kl_weight > 0 and z_agg is not None:
            # Assume standard normal prior: KL(q(z|x) || N(0,I))
            kl_loss = -0.5 * torch.sum(1 + torch.log(z_agg.var(dim=0) + 1e-8) - z_agg.mean(dim=0).pow(2) - z_agg.var(dim=0))
            total_loss += self.kl_weight * kl_loss
            loss_dict["kl_loss"] = kl_loss.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


def test_autoencoder():
    """Test autoencoder architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = COVID19Autoencoder(
        num_slices=64,
        latent_dim=256,
        aggregation="mean"
    ).to(device)

    # Test input
    B, N, H, W = 2, 64, 256, 256
    x = torch.randn(B, N, H, W).to(device)

    # Forward pass
    x_recon, z_agg = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z_agg.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")

    # Compute loss
    criterion = COVID19AutoencoderLoss()
    loss, loss_dict = criterion(x, x_recon, z_agg)

    print(f"Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")

    # Anomaly score
    scores = model.get_anomaly_score(x)
    print(f"Anomaly scores: {scores}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    test_autoencoder()
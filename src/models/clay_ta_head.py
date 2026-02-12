from __future__ import annotations

import torch
import torch.nn as nn


class ClayTAHead(nn.Module):
    def __init__(self, chip_channels: int, patch_size: int, time_dim: int, loc_dim: int, landcover_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        chip_dim = chip_channels * patch_size * patch_size
        meta_dim = time_dim + loc_dim + landcover_dim
        self.backbone = nn.Sequential(
            nn.Linear(chip_dim + meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, time7: torch.Tensor, loc: torch.Tensor, lc: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        feat = torch.cat([x_flat, time7, loc, lc], dim=1)
        return self.backbone(feat).squeeze(1)

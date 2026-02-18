from __future__ import annotations

import torch
import torch.nn as nn
from claymodel.finetune.regression.factory import RegressionEncoder


class ClayCenterRegressor(nn.Module):
    """9x9 patch -> center TA regression (Clay backbone + selectable head)."""

    def __init__(
        self,
        ckpt_path: str,
        freeze_encoder: bool,
        bt_waves_um: list[float],
        rf_waves_um: list[float],
        chip_channels: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        head_type: str = "cross_attn",
        attn_heads: int = 8,
    ):
        super().__init__()
        self.freeze_encoder = bool(freeze_encoder)
        self.encoder = RegressionEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
            ckpt_path=ckpt_path,
        )
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        waves = torch.tensor(bt_waves_um + rf_waves_um, dtype=torch.float32)
        self.register_buffer('waves', waves, persistent=True)

        extra_dim = max(0, int(chip_channels) - 16)
        self.extra_dim = extra_dim
        meta_dim = extra_dim + 7 + 3 + 17
        self.head_type = str(head_type).lower()
        if self.head_type not in {"mlp", "cross_attn"}:
            raise ValueError("head_type must be one of: mlp, cross_attn")

        if self.head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(1024 + meta_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=1024,
                num_heads=int(attn_heads),
                dropout=dropout,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(1024 + 1024, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor, time7: torch.Tensor, loc: torch.Tensor, lc: torch.Tensor) -> torch.Tensor:
        spectral = torch.cat([x[:, 0:10], x[:, 10:16]], dim=1)
        extra = x[:, 16:16 + self.extra_dim]
        c = extra.shape[-1] // 2
        extra_center = torch.nan_to_num(extra[:, :, c, c], nan=0.0, posinf=0.0, neginf=0.0)

        datacube = {
            'pixels': spectral,
            'time': time7[:, 0:4],
            'latlon': torch.cat([loc[:, 0:2], time7[:, 4:6]], dim=1),
            'gsd': torch.tensor([float(loc[0, 2].item())], device=x.device, dtype=x.dtype),
            'waves': self.waves.to(x.device, dtype=x.dtype),
        }
        if self.freeze_encoder:
            with torch.no_grad():
                patches = self.encoder(datacube)
        else:
            patches = self.encoder(datacube)
        emb = patches.mean(dim=1)
        meta = torch.cat([extra_center, time7, loc, lc], dim=1)

        if self.head_type == "mlp":
            feat = torch.cat([emb, meta], dim=1)
            return self.head(feat).squeeze(1)

        q = emb.unsqueeze(1)  # (B,1,1024)
        kv = self.meta_proj(meta).unsqueeze(1)  # (B,1,1024)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        feat = torch.cat([emb, attn_out.squeeze(1)], dim=1)
        return self.head(feat).squeeze(1)

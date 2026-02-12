from __future__ import annotations

import torch
import torch.nn as nn

from claymodel.finetune.regression.factory import RegressionEncoder


class ClayTransferRegressor(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        freeze_encoder: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        bt_waves_um: list[float] | None = None,
        rf_waves_um: list[float] | None = None,
    ):
        super().__init__()
        if not ckpt_path:
            raise ValueError("Clay pretrained ckpt_path is required")
        self.freeze_encoder = freeze_encoder

        # Clay encoder (dynamic embedding + pos/meta encoding)
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
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        bt = bt_waves_um or [3.80, 6.30, 6.90, 7.30, 8.70, 9.60, 10.50, 11.20, 12.30, 13.30]
        rf = rf_waves_um or [0.47, 0.51, 0.64, 0.86, 1.37, 1.60]
        waves = torch.tensor(bt + rf, dtype=torch.float32)  # x channel order: BT10 then RF6
        self.register_buffer("waves", waves, persistent=True)

        # Extra engineered channels/meta for regression head
        # extra = mean(SZA/DEM/LSMASK)=3 + time7=7 + loc3=3 + lc17=17 => 30
        in_dim = 1024 + 30
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, time7: torch.Tensor, loc: torch.Tensor, lc: torch.Tensor) -> torch.Tensor:
        # x: [B, 19, H, W] where 0:10 BT, 10:16 RF, 16:19 extra(SZA,DEM,LSMASK)
        b = x.shape[0]
        spectral = torch.cat([x[:, 0:10], x[:, 10:16]], dim=1)  # [B,16,H,W]
        extra_map = x[:, 16:19].mean(dim=(2, 3))  # [B,3]

        # Clay expects time(4), latlon(4), gsd(1), waves(16)
        time4 = time7[:, 0:4]
        latlon4 = torch.cat([loc[:, 0:2], time7[:, 4:6]], dim=1)
        gsd = torch.tensor([float(loc[0, 2].item())], device=x.device, dtype=x.dtype)
        datacube = {
            "pixels": spectral,
            "time": time4,
            "latlon": latlon4,
            "gsd": gsd,
            "waves": self.waves.to(x.device, dtype=x.dtype),
        }

        if self.freeze_encoder:
            with torch.no_grad():
                patches = self.encoder(datacube)  # [B, L, 1024]
        else:
            patches = self.encoder(datacube)  # [B, L, 1024]
        emb = patches.mean(dim=1)  # [B,1024]
        fused = torch.cat([emb, extra_map, time7, loc, lc], dim=1)  # [B,1054]
        return self.head(fused).squeeze(1)

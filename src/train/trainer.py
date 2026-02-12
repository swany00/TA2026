from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import IndexDataset, PatchFileAwareSampler, PatchPTDataset, ShardDataset
from src.models.clay_ta_head import ClayTAHead
from src.models.clay_transfer import ClayTransferRegressor
from src.models.losses import build_loss
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.loops import run_epoch
from src.utils.io import ensure_dir, load_json


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_train(cfg: dict) -> None:
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    target_stats = load_json(dcfg["target_stats_path"])
    ta_std = float(target_stats.get("TA", {}).get("std", 1.0)) or 1.0
    _seed(int(tcfg["seed"]))
    ensure_dir(tcfg["checkpoint_dir"])
    ensure_dir(tcfg["log_dir"])

    device = tcfg["device"]
    strict_cuda = bool(tcfg.get("strict_cuda", True))
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        msg = "[train][error] CUDA requested but unavailable."
        if strict_cuda:
            raise RuntimeError(msg + " Set up GPU runtime or set train.strict_cuda=false to allow CPU fallback.")
        print(msg + " Falling back to CPU.", flush=True)
        device = "cpu"
    if str(device).startswith("cuda"):
        # Fix default CUDA context to requested device before model/ckpt init.
        torch.cuda.set_device(torch.device(device))
        cur_idx = torch.cuda.current_device()
        cur_name = torch.cuda.get_device_name(cur_idx)
        print(f"[train] cuda current_device={cur_idx} name={cur_name}", flush=True)
        if bool(tcfg.get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    print(f"[train] device={device}", flush=True)

    data_source = str(tcfg.get("data_source", "shard")).lower()
    train_sampler = None
    if data_source == "index":
        if float(tcfg.get("sample_fraction", 1.0)) >= 1.0 and int(tcfg.get("max_samples_per_split", 0)) <= 0:
            print("[train] loading full index rows (this can take a long time)", flush=True)
        train_ds = IndexDataset(cfg, "train")
        val_ds = IndexDataset(cfg, "val")
    elif data_source == "patch_pt":
        train_ds = PatchPTDataset(tcfg["patch_pt_root"], "train")
        val_ds = PatchPTDataset(tcfg["patch_pt_root"], "val")
        train_sampler = PatchFileAwareSampler(
            train_ds,
            seed=int(tcfg.get("seed", 42)),
            shuffle_files=bool(tcfg.get("patch_shuffle_files", True)),
            shuffle_within_file=bool(tcfg.get("patch_shuffle_within_file", True)),
        )
    else:
        train_ds = ShardDataset(tcfg["shard_root"], "train")
        val_ds = ShardDataset(tcfg["shard_root"], "val")
    print(f"[train] dataset sizes: train={len(train_ds)} val={len(val_ds)}")
    num_workers = int(tcfg["num_workers"])
    pin_memory = bool(tcfg.get("pin_memory", True))
    persistent_workers = bool(tcfg.get("persistent_workers", True)) if num_workers > 0 else False
    prefetch_factor = int(tcfg.get("prefetch_factor", 2))
    loader_common = {
        "batch_size": int(tcfg["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_common["prefetch_factor"] = prefetch_factor
    print(
        "[train] dataloader "
        f"num_workers={num_workers} pin_memory={pin_memory} "
        f"persistent_workers={persistent_workers} "
        f"prefetch_factor={prefetch_factor if num_workers > 0 else 'n/a'}",
        flush=True,
    )
    if train_sampler is not None:
        print(
            "[train] patch sampler "
            f"shuffle_files={bool(tcfg.get('patch_shuffle_files', True))} "
            f"shuffle_within_file={bool(tcfg.get('patch_shuffle_within_file', True))}",
            flush=True,
        )
        train_loader = DataLoader(train_ds, shuffle=False, sampler=train_sampler, **loader_common)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_common)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_common)

    model_type = str(mcfg.get("type", "mlp")).lower()
    if model_type == "clay_transfer":
        ckpt_path = str(mcfg.get("clay_ckpt_path", ""))
        model = ClayTransferRegressor(
            ckpt_path=ckpt_path,
            freeze_encoder=bool(mcfg.get("freeze_encoder", True)),
            hidden_dim=int(mcfg.get("hidden_dim", 256)),
            dropout=float(mcfg.get("dropout", 0.1)),
            bt_waves_um=mcfg.get("bt_waves_um"),
            rf_waves_um=mcfg.get("rf_waves_um"),
        ).to(device)
        print(
            f"[train] model=clay_transfer ckpt={ckpt_path} "
            f"freeze_encoder={bool(mcfg.get('freeze_encoder', True))} "
            f"waves={int(model.waves.numel())}",
            flush=True,
        )
    else:
        model = ClayTAHead(
            chip_channels=int(dcfg["chip_channels"]),
            patch_size=int(dcfg["patch_size"]),
            time_dim=int(dcfg["time_dim"]),
            loc_dim=int(dcfg["loc_dim"]),
            landcover_dim=int(dcfg["landcover_dim"]),
            hidden_dim=int(mcfg["hidden_dim"]),
            dropout=float(mcfg["dropout"]),
        ).to(device)
    model_param_device = next(model.parameters()).device
    print(f"[train] model_param_device={model_param_device}", flush=True)
    criterion = build_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    use_amp = bool(tcfg.get("amp", True)) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cuda_prefetch_batches = int(tcfg.get("cuda_prefetch_batches", 0))
    print(
        f"[train] precision amp={use_amp} tf32={bool(tcfg.get('tf32', True))} "
        f"cuda_prefetch_batches={cuda_prefetch_batches}",
        flush=True,
    )

    start_epoch, best_val = load_checkpoint(tcfg["checkpoint_dir"], model, optimizer, scheduler)
    total_epochs = int(tcfg["epochs"])
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss, tr_rmse_norm = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            epochs=total_epochs,
            use_amp=use_amp,
            scaler=scaler,
            prefetch_batches=cuda_prefetch_batches,
        )
        va_loss, va_rmse_norm = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            epoch=epoch,
            epochs=total_epochs,
            use_amp=use_amp,
            scaler=scaler,
            prefetch_batches=cuda_prefetch_batches,
        )
        scheduler.step(va_loss)
        tr_rmse_k = tr_rmse_norm * ta_std
        va_rmse_k = va_rmse_norm * ta_std
        print(
            f"[train] epoch={epoch+1} "
            f"train_loss={tr_loss:.6f} val_loss={va_loss:.6f} "
            f"train_rmse_norm={tr_rmse_norm:.6f} val_rmse_norm={va_rmse_norm:.6f} "
            f"train_rmse_K={tr_rmse_k:.6f} val_rmse_K={va_rmse_k:.6f}"
        )
        is_best = va_loss < best_val
        if va_loss < best_val:
            best_val = va_loss
        save_checkpoint(tcfg["checkpoint_dir"], model, optimizer, scheduler, epoch, best_val, is_best=is_best)
    print(f"[train] best_val={best_val:.6f}")

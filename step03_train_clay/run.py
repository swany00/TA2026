#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.dataset_patch_pt import PatchPTDataset
from shared.io import ensure_dir, load_json, load_yaml
from shared.model_clay import ClayCenterRegressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    return p.parse_args()


def rmse(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - y) ** 2)).item())


def run():
    args = parse_args()
    cfg = load_yaml(args.config)
    t = cfg['train']
    m = cfg['model']
    d = cfg['data']

    device = str(t.get('device', 'cuda:0'))
    cpu_threads = int(t.get('cpu_threads', 1))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))

    file_cache_size = int(t.get('file_cache_size', 8))
    ds_train = PatchPTDataset(t['patch_pt_root'], 'train', cache_size=file_cache_size)
    ds_val = PatchPTDataset(t['patch_pt_root'], 'val', cache_size=file_cache_size)

    num_workers = int(t.get('num_workers', 8))
    pin_memory = bool(t.get('pin_memory', True))
    persistent_workers = bool(t.get('persistent_workers', True)) if num_workers > 0 else False
    prefetch_factor = int(t.get('prefetch_factor', 2))
    loader_common = {
        'batch_size': int(t['batch_size']),
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
    }
    if num_workers > 0:
        loader_common['prefetch_factor'] = prefetch_factor
    print(
        f"[train] cpu_threads={cpu_threads} file_cache_size={file_cache_size}\n"
        f"[train] dataloader num_workers={num_workers} pin_memory={pin_memory} "
        f"persistent_workers={persistent_workers} prefetch_factor={prefetch_factor if num_workers > 0 else 'n/a'}",
        flush=True,
    )

    train_loader = DataLoader(
        ds_train,
        shuffle=False,
        **loader_common,
    )

    val_n = int(t.get('val_random_samples', 0))
    if val_n > 0 and len(ds_val) > val_n:
        rng = np.random.RandomState(int(t.get('seed', 42)))
        idx = rng.choice(len(ds_val), size=val_n, replace=False).tolist()
        ds_val_eval = Subset(ds_val, idx)
    else:
        ds_val_eval = ds_val

    val_loader = DataLoader(
        ds_val_eval,
        shuffle=False,
        **loader_common,
    )

    chip_channels = int(ds_train[0]['x'].shape[0])
    model = ClayCenterRegressor(
        ckpt_path=m['clay_ckpt_path'],
        freeze_encoder=bool(m.get('freeze_encoder', True)),
        bt_waves_um=list(m['bt_waves_um']),
        rf_waves_um=list(m['rf_waves_um']),
        chip_channels=chip_channels,
        hidden_dim=int(m.get('hidden_dim', 256)),
        dropout=float(m.get('dropout', 0.1)),
        head_type=str(m.get('head_type', 'cross_attn')),
        attn_heads=int(m.get('attn_heads', 8)),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=float(t['lr']), weight_decay=float(t.get('weight_decay', 1e-5)))
    crit = torch.nn.MSELoss()

    ta_std = float(load_json(d['target_stats_path'])['TA']['std'])
    ckpt_dir = Path(t['checkpoint_dir'])
    ensure_dir(ckpt_dir)
    best = float('inf')

    for ep in range(int(t['epochs'])):
        model.train()
        tr_loss, tr_sse, tr_n = 0.0, 0.0, 0
        p = tqdm(train_loader, desc=f"train {ep+1}/{int(t['epochs'])}", unit='batch')
        for b in p:
            x = b['x'].to(device, non_blocking=True)
            y = b['y'].to(device, non_blocking=True)
            t7 = b['time7'].to(device, non_blocking=True)
            loc = b['loc'].to(device, non_blocking=True)
            lc = b['landcover_onehot'].to(device, non_blocking=True)

            valid = torch.isfinite(x[:, :16]).all(dim=(1,2,3)) & torch.isfinite(y)
            if not bool(valid.any()):
                continue
            x, y, t7, loc, lc = x[valid], y[valid], t7[valid], loc[valid], lc[valid]

            pred = model(x, t7, loc, lc)
            loss = crit(pred, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bs = x.size(0)
            tr_loss += float(loss.item()) * bs
            tr_sse += float(torch.sum((pred.detach() - y) ** 2).item())
            tr_n += bs
            p.set_postfix(loss=f"{loss.item():.4f}", rmse=f"{(tr_sse/max(1,tr_n))**0.5:.4f}", krmse=f"{((tr_sse/max(1,tr_n))**0.5*ta_std):.4f}")

        model.eval()
        va_loss, va_sse, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in tqdm(val_loader, desc='val', unit='batch'):
                x = b['x'].to(device, non_blocking=True)
                y = b['y'].to(device, non_blocking=True)
                t7 = b['time7'].to(device, non_blocking=True)
                loc = b['loc'].to(device, non_blocking=True)
                lc = b['landcover_onehot'].to(device, non_blocking=True)

                valid = torch.isfinite(x[:, :16]).all(dim=(1,2,3)) & torch.isfinite(y)
                if not bool(valid.any()):
                    continue
                x, y, t7, loc, lc = x[valid], y[valid], t7[valid], loc[valid], lc[valid]

                pred = model(x, t7, loc, lc)
                loss = crit(pred, y)
                bs = x.size(0)
                va_loss += float(loss.item()) * bs
                va_sse += float(torch.sum((pred - y) ** 2).item())
                va_n += bs

        tr_rmse = (tr_sse / max(1, tr_n)) ** 0.5
        va_rmse = (va_sse / max(1, va_n)) ** 0.5
        tr_avg = tr_loss / max(1, tr_n)
        va_avg = va_loss / max(1, va_n)
        print(
            f"[epoch {ep+1}] train_loss={tr_avg:.6f} val_loss={va_avg:.6f} "
            f"train_rmse={tr_rmse:.6f} val_rmse={va_rmse:.6f} "
            f"train_krmse={tr_rmse*ta_std:.6f} val_krmse={va_rmse*ta_std:.6f}",
            flush=True,
        )

        torch.save({'model': model.state_dict(), 'epoch': ep, 'best_val': best}, ckpt_dir / 'current.pt')
        if va_avg < best:
            best = va_avg
            torch.save({'model': model.state_dict(), 'epoch': ep, 'best_val': best}, ckpt_dir / 'best.pt')


if __name__ == '__main__':
    run()

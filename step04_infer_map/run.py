#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.io import ensure_dir, load_json, load_yaml
from shared.model_clay import ClayCenterRegressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    return p.parse_args()


def load_alpha(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == '.npy':
        a = np.load(str(p)).astype(np.float32)
    else:
        import xarray as xr
        ds = xr.open_dataset(str(p))
        a = ds['embedding'].values.astype(np.float32)
        ds.close()
    if a.shape[1:] != (900, 900):
        a = a.transpose(2, 0, 1)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def run():
    args = parse_args()
    cfg = load_yaml(args.config)
    i = cfg['infer']
    m = cfg['model']
    s = load_json(cfg['stats']['input'])
    tstats = load_json(cfg['stats']['target'])

    device = i.get('device', 'cuda:0')
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))

    bt = np.load(i['bt']).astype(np.float32)
    rf = np.load(i['rf']).astype(np.float32)
    sza = np.load(i['sza']).astype(np.float32)
    dem = np.fromfile(i['dem_bin'], dtype=np.int16).reshape(900, 900).astype(np.float32)
    dem[dem < 0] = 0.0
    lsm = np.fromfile(i['lsmask_bin'], dtype=np.uint16).reshape(900, 900).astype(np.float32)
    alpha = load_alpha(i['alpha'])

    bt_mean = np.array([float(s.get(f'bt_{k:02d}', {}).get('mean', 0.0)) for k in range(10)], dtype=np.float32)
    bt_std = np.array([float(s.get(f'bt_{k:02d}', {}).get('std', 1.0)) for k in range(10)], dtype=np.float32)
    bt_std[bt_std == 0] = 1.0
    rf_mean = np.array([float(s.get(f'reflectance_{k:02d}', {}).get('mean', 0.0)) for k in range(6)], dtype=np.float32)
    rf_std = np.array([float(s.get(f'reflectance_{k:02d}', {}).get('std', 1.0)) for k in range(6)], dtype=np.float32)
    rf_std[rf_std == 0] = 1.0
    sza_mean, sza_std = float(s.get('sza', {}).get('mean', 0.0)), float(s.get('sza', {}).get('std', 1.0)) or 1.0
    dem_mean, dem_std = float(s.get('dem', {}).get('mean', 0.0)), float(s.get('dem', {}).get('std', 1.0)) or 1.0

    model = ClayCenterRegressor(
        ckpt_path=m['clay_ckpt_path'],
        freeze_encoder=bool(m.get('freeze_encoder', False)),
        bt_waves_um=list(m['bt_waves_um']),
        rf_waves_um=list(m['rf_waves_um']),
        chip_channels=83,
        hidden_dim=int(m.get('hidden_dim', 256)),
        dropout=float(m.get('dropout', 0.1)),
    ).to(device).eval()
    state = torch.load(i['checkpoint'], map_location='cpu')
    model.load_state_dict(state['model'] if 'model' in state else state, strict=False)

    ps = 9
    h = ps // 2
    ys, xs = np.mgrid[h:900-h, h:900-h]
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    y0, x0 = ys - h, xs - h

    bt_w = sliding_window_view(bt, (ps, ps), axis=(1, 2))
    rf_w = sliding_window_view(rf, (ps, ps), axis=(1, 2))
    sza_w = sliding_window_view(sza, (ps, ps), axis=(0, 1))
    dem_w = sliding_window_view(dem, (ps, ps), axis=(0, 1))
    lsm_w = sliding_window_view(lsm, (ps, ps), axis=(0, 1))
    a_w = sliding_window_view(alpha, (ps, ps), axis=(1, 2))

    ta_mean = float(tstats['TA']['mean'])
    ta_std = float(tstats['TA']['std'])

    pred = np.full((900, 900), np.nan, dtype=np.float32)
    bs = int(i.get('batch_size', 8192))
    t7 = np.zeros((1, 7), dtype=np.float32)
    loc = np.zeros((1, 3), dtype=np.float32)
    lc = np.zeros((1, 17), dtype=np.float32)

    for st in tqdm(range(0, len(xs), bs), desc='infer', unit='batch'):
        ed = min(len(xs), st + bs)
        yb, xb = y0[st:ed], x0[st:ed]
        yc, xc = ys[st:ed], xs[st:ed]

        bt_p = bt_w[:, yb, xb, :, :].transpose(1, 0, 2, 3)
        rf_p = rf_w[:, yb, xb, :, :].transpose(1, 0, 2, 3)
        sza_p = sza_w[yb, xb, :, :]
        dem_p = dem_w[yb, xb, :, :]
        lsm_p = lsm_w[yb, xb, :, :]
        a_p = a_w[:, yb, xb, :, :].transpose(1, 0, 2, 3)

        x = np.concatenate([
            (bt_p - bt_mean[None, :, None, None]) / bt_std[None, :, None, None],
            (rf_p - rf_mean[None, :, None, None]) / rf_std[None, :, None, None],
            ((sza_p - sza_mean) / sza_std)[:, None],
            ((dem_p - dem_mean) / dem_std)[:, None],
            lsm_p[:, None],
            a_p,
        ], axis=1).astype(np.float32)

        t = np.repeat(t7, len(x), axis=0)
        l = np.repeat(loc, len(x), axis=0)
        c = np.repeat(lc, len(x), axis=0)

        with torch.no_grad():
            y_n = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(t).to(device),
                torch.from_numpy(l).to(device),
                torch.from_numpy(c).to(device),
            ).float().cpu().numpy()
        y_k = y_n * ta_std + ta_mean
        pred[yc, xc] = y_k.astype(np.float32)

    ensure_dir(i['out_dir'])
    out_npy = Path(i['out_dir']) / f"{i['name']}_pred_k.npy"
    np.save(out_npy, pred)

    # simple png
    a = pred.copy()
    vmin, vmax = np.nanmin(a), np.nanmax(a)
    z = np.clip((a - vmin) / max(1e-6, (vmax - vmin)), 0, 1)
    img = (z * 255).astype(np.uint8)
    Image.fromarray(img).save(Path(i['out_dir']) / f"{i['name']}_pred_k.png")
    print(f'[step04] saved: {out_npy}')


if __name__ == '__main__':
    run()

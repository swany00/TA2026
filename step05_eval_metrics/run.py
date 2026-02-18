#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.io import load_yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    return p.parse_args()


def run():
    args = parse_args()
    cfg = load_yaml(args.config)
    y_true = np.load(cfg['y_true']).astype(np.float32)
    y_pred = np.load(cfg['y_pred']).astype(np.float32)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        raise RuntimeError('No valid samples for evaluation')

    t = y_true[valid]
    p = y_pred[valid]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    bias = float(np.mean(p - t))

    print(f"MAE={mae:.4f} RMSE={rmse:.4f} Bias={bias:.4f} N={int(valid.sum())}")


if __name__ == '__main__':
    run()

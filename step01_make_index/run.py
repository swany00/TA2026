#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.io import ensure_dir, load_yaml


COLS = ["timestamp_utc", "stn", "lat", "lon", "pixel_x", "pixel_y", "ta_k"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return p.parse_args()


def _split_of_year(year: int, cfg: dict) -> str:
    if year in set(cfg["split"]["train_years"]):
        return "train"
    if year in set(cfg["split"]["val_years"]):
        return "val"
    if year in set(cfg["split"]["test_years"]):
        return "test"
    return "skip"


def _process_month(args: Tuple[str, str, dict, str]) -> Dict[str, object]:
    month_path_str, month_key, cfg, tmp_dir = args
    month_path = Path(month_path_str)
    arr = np.load(str(month_path), allow_pickle=True)
    # [timestamp_utc, stn, lat, lon, pixel_x, pixel_y, ta_k]
    ts = arr[:, 0].astype(str)
    stn = arr[:, 1]
    lat = arr[:, 2].astype(np.float32)
    lon = arr[:, 3].astype(np.float32)
    px = arr[:, 4].astype(np.float32)
    py = arr[:, 5].astype(np.float32)
    ta = arr[:, 6].astype(np.float32)

    patch = int(cfg["data"]["patch_size"])
    r = patch // 2
    h = int(cfg["data"]["grid_h"])
    w = int(cfg["data"]["grid_w"])
    ta_k_min = cfg["data"].get("ta_k_min", None)
    ta_k_max = cfg["data"].get("ta_k_max", None)

    mask = (
        np.isfinite(px)
        & np.isfinite(py)
        & np.isfinite(ta)
        & (px >= r)
        & (py >= r)
        & (px <= (w - 1 - r))
        & (py <= (h - 1 - r))
    )
    if ta_k_min is not None:
        mask = mask & (ta >= float(ta_k_min))
    if ta_k_max is not None:
        mask = mask & (ta <= float(ta_k_max))
    if not bool(mask.any()):
        return {"month": month_key, "counts": {"train": 0, "val": 0, "test": 0}, "files": {}}

    ts = ts[mask]
    stn = stn[mask]
    lat = lat[mask]
    lon = lon[mask]
    px = px[mask].astype(np.int32)
    py = py[mask].astype(np.int32)
    ta = ta[mask]

    year = int(month_key[:4])
    split = _split_of_year(year, cfg)
    if split == "skip":
        return {"month": month_key, "counts": {"train": 0, "val": 0, "test": 0}, "files": {}}

    df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "stn": stn,
            "lat": lat,
            "lon": lon,
            "pixel_x": px,
            "pixel_y": py,
            "ta_k": ta,
        }
    )
    out = Path(tmp_dir) / f"{split}_{month_key}.csv"
    df.to_csv(out, index=False)
    return {
        "month": month_key,
        "counts": {"train": int(len(df)) if split == "train" else 0, "val": int(len(df)) if split == "val" else 0, "test": int(len(df)) if split == "test" else 0},
        "files": {split: str(out)},
    }


def _concat_csv(files: List[Path], out_csv: Path) -> int:
    if not files:
        pd.DataFrame(columns=COLS).to_csv(out_csv, index=False)
        return 0
    ensure_dir(out_csv.parent)
    first = True
    total = 0
    with out_csv.open("w", encoding="utf-8") as fw:
        for fp in files:
            with fp.open("r", encoding="utf-8") as fr:
                header = fr.readline()
                if first:
                    fw.write(header)
                    first = False
                for line in fr:
                    fw.write(line)
                    total += 1
    return total


def run():
    args = parse_args()
    cfg = load_yaml(args.config)

    label_root = Path(cfg["data"]["ta_label_root"])
    out_root = Path(cfg["output"]["root"])
    tmp_dir = out_root / "tmp"
    ensure_dir(tmp_dir)

    month_files = sorted(label_root.glob("*/*.npy"))
    if not month_files:
        raise RuntimeError(f"No month label npy files under: {label_root}")

    jobs = [(str(p), p.stem, cfg, str(tmp_dir)) for p in month_files]
    num_workers = int(cfg["runtime"].get("num_workers", 16))

    split_files: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_month, j) for j in jobs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="build_index", unit="month"):
            r = f.result()
            for s in ["train", "val", "test"]:
                counts[s] += int(r["counts"][s])
                fp = r["files"].get(s)
                if fp:
                    split_files[s].append(Path(fp))

    split_files["train"].sort()
    split_files["val"].sort()
    split_files["test"].sort()

    train_csv = out_root / "index_train.csv"
    val_csv = out_root / "index_val.csv"
    test_csv = out_root / "index_test.csv"
    total_csv = out_root / "index.csv"

    n_train = _concat_csv(split_files["train"], train_csv)
    n_val = _concat_csv(split_files["val"], val_csv)
    n_test = _concat_csv(split_files["test"], test_csv)

    with total_csv.open("w", encoding="utf-8") as fw:
        first = True
        for p in [train_csv, val_csv, test_csv]:
            with p.open("r", encoding="utf-8") as fr:
                header = fr.readline()
                if first:
                    fw.write(header)
                    first = False
                for line in fr:
                    fw.write(line)

    print(f"[step01] saved: {train_csv} rows={n_train}")
    print(f"[step01] saved: {val_csv} rows={n_val}")
    print(f"[step01] saved: {test_csv} rows={n_test}")
    print(f"[step01] saved: {total_csv} rows={n_train + n_val + n_test}")


if __name__ == "__main__":
    run()

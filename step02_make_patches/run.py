#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.io import ensure_dir, load_json, load_yaml


G = {
    "cfg": None,
    "stats_in": None,
    "stats_ta": None,
    "dem": None,
    "lsm": None,
    "lat": None,
    "lon": None,
    "landcover_by_year": {},
    "alpha_by_year": {},
    "cache_bt": {},
    "cache_rf": {},
    "cache_sza": {},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return p.parse_args()


def _build_paths(data_cfg: dict, ts: str) -> Dict[str, str]:
    yyyymm = ts[:6]
    day = ts[6:8]
    year = ts[:4]
    month = ts[4:6]
    out = {
        "bt": f"{data_cfg['bt_root']}/{yyyymm}/{day}/gk2a_ami_le1b_all_ch_ko020lc_{ts}_bt.npy",
        "rf": f"{data_cfg['reflectance_root']}/{yyyymm}/{day}/gk2a_ami_le1b_all_ch_ko020lc_{ts}_reflectance.npy",
        "sza": f"{data_cfg['sza_root']}/{yyyymm}/{day}/sza_map_{ts}.npy",
    }
    if data_cfg.get("nlsd_ir_root"):
        out["nlsd_ir"] = f"{data_cfg['nlsd_ir_root']}/{year}/{month}/gk2a_ami_le1b_all_ch_ko020lc_{ts}_NLSD.npy"
    if data_cfg.get("nlsd_vi_root"):
        out["nlsd_vi"] = f"{data_cfg['nlsd_vi_root']}/{year}/{month}/gk2a_ami_le1b_all_ch_ko020lc_{ts}_NLSD.npy"
    return out


def _time7(ts: str, sza_center: float) -> np.ndarray:
    dt = datetime.strptime(ts, "%Y%m%d%H%M")
    doy = dt.timetuple().tm_yday
    hour = dt.hour + (dt.minute / 60.0)
    month = dt.month
    return np.asarray(
        [
            np.cos(2.0 * np.pi * doy / 366.0),
            np.sin(2.0 * np.pi * doy / 366.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * month / 12.0),
            np.sin(2.0 * np.pi * month / 12.0),
            sza_center,
        ],
        dtype=np.float32,
    )


def _load_landcover(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.int32)
    return np.clip(arr, 0, 16)


def _init_worker(cfg: dict):
    G["cfg"] = cfg
    G["stats_in"] = load_json(cfg["stats"]["input"])
    G["stats_ta"] = load_json(cfg["stats"]["target"])

    dem = np.fromfile(cfg["data"]["dem_bin"], dtype=np.int16).reshape(900, 900).astype(np.float32)
    # VER3 규칙: DEM 음수는 0으로 클리핑
    dem = np.where(dem < 0.0, 0.0, dem).astype(np.float32)
    dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
    G["dem"] = dem
    G["lsm"] = np.fromfile(cfg["data"]["lsmask_bin"], dtype=np.uint16).reshape(900, 900).astype(np.float32)

    ds = xr.open_dataset(cfg["data"]["latlon_nc"])
    G["lat"] = ds["lat"].values.astype(np.float32)
    G["lon"] = ds["lon"].values.astype(np.float32)
    ds.close()

    lc_dir = Path(cfg["data"]["landcover_dir"])
    for y in cfg["data"]["landcover_years"]:
        p = lc_dir / f"MCD12C1_{int(y)}_GK2A_label.png"
        if p.exists():
            G["landcover_by_year"][int(y)] = _load_landcover(p)

    # thread_pool 안정성: netCDF/xarray 파일 오픈을 멀티스레드에서 하지 않도록
    # thread_pool 모드에서만 초기화 시점에 연도별 AlphaEarth를 순차 preload
    if str(cfg.get("runtime", {}).get("executor_mode", "process_pool")).lower() == "thread_pool":
        for y in cfg["data"].get("alpha_years", []):
            yy = int(y)
            if yy in G["alpha_by_year"]:
                continue
            p = Path(cfg["data"]["alphaearth_root"]) / f"KO_Embedding_{yy}_2km_on_KOgrid.nc"
            if not p.exists():
                continue
            ds = xr.open_dataset(p)
            arr = ds["embedding"].values.astype(np.float32)
            ds.close()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            G["alpha_by_year"][yy] = arr


def _get_landcover_map(year: int) -> np.ndarray:
    if year in G["landcover_by_year"]:
        return G["landcover_by_year"][year]
    if not G["landcover_by_year"]:
        raise RuntimeError("No landcover png loaded")
    nearest = min(G["landcover_by_year"].keys(), key=lambda y: abs(y - year))
    return G["landcover_by_year"][nearest]


def _get_alpha(year: int) -> np.ndarray:
    if year in G["alpha_by_year"]:
        return G["alpha_by_year"][year]
    root = Path(G["cfg"]["data"]["alphaearth_root"])

    # strict: exact year file only
    p = root / f"KO_Embedding_{year}_2km_on_KOgrid.nc"
    if p.exists():
        ds = xr.open_dataset(p)
        arr = ds["embedding"].values.astype(np.float32)
        ds.close()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        G["alpha_by_year"][year] = arr
        return arr

    raise RuntimeError(
        f"AlphaEarth file not found for exact year={year}: "
        f"{root / f'KO_Embedding_{year}_2km_on_KOgrid.nc'}"
    )


def _cache_get(cache: dict, path: str, max_items: int) -> np.ndarray:
    if path in cache:
        return cache[path]
    arr = np.load(path).astype(np.float32)
    cache[path] = arr
    if len(cache) > max_items:
        cache.pop(next(iter(cache)))
    return arr


def _norm_ch(ch: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std2 = std.copy()
    std2[std2 == 0] = 1.0
    return (ch - mean[:, None, None]) / std2[:, None, None]


def _norm_2d(ch: np.ndarray, mean: float, std: float) -> np.ndarray:
    s = 1.0 if std == 0 else std
    return (ch - mean) / s


def _make_onehot(v: int, n: int = 17) -> np.ndarray:
    o = np.zeros((n,), dtype=np.float32)
    o[int(v)] = 1.0
    return o


def _build_pt_chunk(task: Tuple[str, int, List[Tuple]]) -> Tuple[str, int, int, int]:
    split, chunk_id, rows = task
    cfg = G["cfg"]
    patch = int(cfg["data"]["patch_size"])
    r = patch // 2
    data_cfg = cfg["data"]
    max_cache = int(cfg["runtime"].get("file_cache_items", 8))
    include_nlsd = bool(cfg["data"].get("include_nlsd", False))

    stats_in = G["stats_in"]
    bt_mean = np.asarray([float(stats_in.get(f"bt_{k:02d}", {}).get("mean", 0.0)) for k in range(10)], dtype=np.float32)
    bt_std = np.asarray([float(stats_in.get(f"bt_{k:02d}", {}).get("std", 1.0)) for k in range(10)], dtype=np.float32)
    rf_mean = np.asarray([float(stats_in.get(f"reflectance_{k:02d}", {}).get("mean", 0.0)) for k in range(6)], dtype=np.float32)
    rf_std = np.asarray([float(stats_in.get(f"reflectance_{k:02d}", {}).get("std", 1.0)) for k in range(6)], dtype=np.float32)
    sza_mean = float(stats_in.get("sza", {}).get("mean", 0.0))
    sza_std = float(stats_in.get("sza", {}).get("std", 1.0))
    dem_mean = float(stats_in.get("dem", {}).get("mean", 0.0))
    dem_std = float(stats_in.get("dem", {}).get("std", 1.0))
    ta_mean = float(G["stats_ta"]["TA"]["mean"])
    ta_std = float(G["stats_ta"]["TA"]["std"]) or 1.0
    ta_k_min = cfg["data"].get("ta_k_min", None)
    ta_k_max = cfg["data"].get("ta_k_max", None)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    t7s: List[np.ndarray] = []
    locs: List[np.ndarray] = []
    lcs: List[np.ndarray] = []
    metas: List[dict] = []
    dropped = 0

    for (ts, stn, lat, lon, px, py, ta_k) in rows:
        ta_v = float(ta_k)
        if not np.isfinite(ta_v):
            dropped += 1
            continue
        if ta_k_min is not None and ta_v < float(ta_k_min):
            dropped += 1
            continue
        if ta_k_max is not None and ta_v > float(ta_k_max):
            dropped += 1
            continue
        px_i = int(px)
        py_i = int(py)
        paths = _build_paths(data_cfg, ts)
        if not (Path(paths["bt"]).exists() and Path(paths["rf"]).exists() and Path(paths["sza"]).exists()):
            dropped += 1
            continue
        if include_nlsd and not (Path(paths["nlsd_ir"]).exists() and Path(paths["nlsd_vi"]).exists()):
            dropped += 1
            continue

        try:
            bt = _cache_get(G["cache_bt"], paths["bt"], max_cache)
            rf = _cache_get(G["cache_rf"], paths["rf"], max_cache)
            sza = _cache_get(G["cache_sza"], paths["sza"], max_cache)
        except Exception:
            dropped += 1
            continue

        y0, y1 = py_i - r, py_i + r + 1
        x0, x1 = px_i - r, px_i + r + 1
        bt_p = bt[:, y0:y1, x0:x1]
        rf_p = rf[:, y0:y1, x0:x1]
        sza_p = sza[y0:y1, x0:x1]
        dem_p = G["dem"][y0:y1, x0:x1]
        lsm_p = G["lsm"][y0:y1, x0:x1]
        if bt_p.shape != (10, patch, patch) or rf_p.shape != (6, patch, patch):
            dropped += 1
            continue

        year = int(ts[:4])
        alpha = _get_alpha(year)
        alpha_p = alpha[:, y0:y1, x0:x1]
        if alpha_p.shape[1:] != (patch, patch):
            dropped += 1
            continue

        chans = [
            _norm_ch(bt_p, bt_mean, bt_std),
            _norm_ch(rf_p, rf_mean, rf_std),
            _norm_2d(sza_p, sza_mean, sza_std)[None, :, :],
            _norm_2d(dem_p, dem_mean, dem_std)[None, :, :],
            lsm_p[None, :, :].astype(np.float32),
            alpha_p.astype(np.float32),
        ]

        if include_nlsd:
            ir = np.load(paths["nlsd_ir"]).astype(np.float32)[y0:y1, x0:x1]
            vi = np.load(paths["nlsd_vi"]).astype(np.float32)[y0:y1, x0:x1]
            chans.extend([ir[None, :, :], vi[None, :, :]])

        x = np.concatenate(chans, axis=0).astype(np.float32)
        if not np.isfinite(x[:16]).all():
            dropped += 1
            continue

        lc_map = _get_landcover_map(year)
        lc_cls = int(np.clip(lc_map[py_i, px_i], 0, 16))
        sza_center = float(sza[py_i, px_i])
        t7 = _time7(ts, sza_center)
        y_norm = np.float32((ta_v - ta_mean) / ta_std)

        xs.append(x)
        ys.append(y_norm)
        t7s.append(t7)
        locs.append(np.asarray([float(lat), float(lon), float(data_cfg["gsd_m"])], dtype=np.float32))
        lcs.append(_make_onehot(lc_cls))
        metas.append({"timestamp_utc": ts, "stn": int(stn), "pixel_x": px_i, "pixel_y": py_i})

    out_root = Path(cfg["output"]["patch_root"])
    ensure_dir(out_root)
    out_pt = out_root / f"{split}_chunk_{chunk_id:05d}.pt"

    if xs:
        data = {
            "x": torch.from_numpy(np.stack(xs, axis=0)),
            "y": torch.from_numpy(np.asarray(ys, dtype=np.float32)),
            "time7": torch.from_numpy(np.stack(t7s, axis=0)),
            "loc": torch.from_numpy(np.stack(locs, axis=0)),
            "landcover_onehot": torch.from_numpy(np.stack(lcs, axis=0)),
            "meta": metas,
        }
        torch.save(data, out_pt)
        n_saved = len(xs)
    else:
        data = {
            "x": torch.empty((0, 83 + (2 if include_nlsd else 0), patch, patch), dtype=torch.float32),
            "y": torch.empty((0,), dtype=torch.float32),
            "time7": torch.empty((0, 7), dtype=torch.float32),
            "loc": torch.empty((0, 3), dtype=torch.float32),
            "landcover_onehot": torch.empty((0, 17), dtype=torch.float32),
            "meta": [],
        }
        torch.save(data, out_pt)
        n_saved = 0

    return split, chunk_id, n_saved, dropped


def _iter_rows(df: pd.DataFrame) -> List[Tuple]:
    return list(
        zip(
            df["timestamp_utc"].astype(str).tolist(),
            df["stn"].tolist(),
            df["lat"].tolist(),
            df["lon"].tolist(),
            df["pixel_x"].tolist(),
            df["pixel_y"].tolist(),
            df["ta_k"].tolist(),
        )
    )


def run():
    args = parse_args()
    cfg = load_yaml(args.config)
    ensure_dir(Path(cfg["output"]["patch_root"]))

    workers = int(cfg["runtime"].get("num_workers", 16))
    executor_mode = str(cfg["runtime"].get("executor_mode", "thread_pool")).strip().lower()
    inflight = int(cfg["runtime"].get("max_inflight_tasks", workers * 2))
    chunk_rows = int(cfg["runtime"].get("chunk_rows", 100000))
    skip_existing = bool(cfg["runtime"].get("skip_existing", True))

    split_to_index = {
        "train": Path(cfg["input"]["index_train_csv"]),
        "val": Path(cfg["input"]["index_val_csv"]),
        "test": Path(cfg["input"]["index_test_csv"]),
    }

    task_total = 0
    for split, p in split_to_index.items():
        if not p.exists():
            continue
        n = 0
        for _ in pd.read_csv(p, chunksize=chunk_rows):
            n += 1
        task_total += n

    pbar = tqdm(total=task_total, desc="build_patches_pt", unit="chunk")
    chunk_counter = {"train": 0, "val": 0, "test": 0}
    saved_counter = {"train": 0, "val": 0, "test": 0}
    drop_counter = {"train": 0, "val": 0, "test": 0}

    if executor_mode not in {"thread_pool", "process_pool"}:
        raise ValueError("runtime.executor_mode must be one of: thread_pool, process_pool")

    if executor_mode == "thread_pool":
        # thread_pool: 정적데이터를 프로세스 전체에서 1회 로드, 청크 직렬화 오버헤드 제거
        _init_worker(cfg)
        ex = ThreadPoolExecutor(max_workers=workers)
    else:
        ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg,))

    print(
        f"[step02] mode={executor_mode} workers={workers} inflight={inflight} chunk_rows={chunk_rows}",
        flush=True,
    )

    with ex:
        futures = set()

        def submit_task(split: str, rows: List[Tuple], cid: int):
            out_pt = Path(cfg["output"]["patch_root"]) / f"{split}_chunk_{cid:05d}.pt"
            if skip_existing and out_pt.exists():
                pbar.update(1)
                return
            futures.add(ex.submit(_build_pt_chunk, (split, cid, rows)))

        for split, csv_path in split_to_index.items():
            if not csv_path.exists():
                continue
            for cdf in pd.read_csv(csv_path, chunksize=chunk_rows):
                cid = chunk_counter[split]
                chunk_counter[split] += 1
                submit_task(split, _iter_rows(cdf), cid)

                while len(futures) >= inflight:
                    done, pending = wait(futures, return_when=FIRST_COMPLETED)
                    futures = pending
                    for d in done:
                        s, _, ns, nd = d.result()
                        saved_counter[s] += int(ns)
                        drop_counter[s] += int(nd)
                        pbar.update(1)

        while futures:
            done, pending = wait(futures, return_when=FIRST_COMPLETED)
            futures = pending
            for d in done:
                s, _, ns, nd = d.result()
                saved_counter[s] += int(ns)
                drop_counter[s] += int(nd)
                pbar.update(1)

    pbar.close()

    print(
        "[step02] saved_samples "
        f"train={saved_counter['train']} val={saved_counter['val']} test={saved_counter['test']}"
    )
    print(
        "[step02] dropped_samples "
        f"train={drop_counter['train']} val={drop_counter['val']} test={drop_counter['test']}"
    )


if __name__ == "__main__":
    run()

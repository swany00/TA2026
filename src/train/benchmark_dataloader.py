from __future__ import annotations

import argparse
import gc
import time

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PatchFileAwareSampler, PatchPTDataset
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark patch_pt dataloader throughput")
    p.add_argument("--config", required=True, help="Path to train yaml")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--workers", default="2,4,8,16")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default="cpu", help="cpu or cuda:idx")
    return p.parse_args()


def run_once(
    ds: PatchPTDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
    shuffle_files: bool,
    shuffle_within_file: bool,
    warmup: int,
    steps: int,
    device: str,
) -> tuple[float, float]:
    sampler = PatchFileAwareSampler(
        ds,
        seed=seed,
        shuffle_files=shuffle_files,
        shuffle_within_file=shuffle_within_file,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "sampler": sampler,
        "shuffle": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(ds, **loader_kwargs)

    total_steps = warmup + steps
    seen = 0
    start = None
    for i, b in enumerate(loader):
        if device.startswith("cuda"):
            _ = b["x"].to(device, non_blocking=True)
            _ = b["y"].to(device, non_blocking=True)
            _ = b["time7"].to(device, non_blocking=True)
            _ = b["loc"].to(device, non_blocking=True)
            _ = b["landcover_onehot"].to(device, non_blocking=True)
        if i == warmup:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            seen = 0
        if i >= warmup:
            seen += int(b["x"].shape[0])
        if i + 1 >= total_steps:
            break

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = max(1e-9, time.perf_counter() - (start or time.perf_counter()))
    batches_per_sec = steps / elapsed
    samples_per_sec = seen / elapsed

    del loader
    gc.collect()
    return batches_per_sec, samples_per_sec


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    split = args.split
    root = str(tcfg["patch_pt_root"])
    batch_size = int(tcfg["batch_size"])
    pin_memory = bool(tcfg.get("pin_memory", True))
    prefetch_factor = int(tcfg.get("prefetch_factor", 2))
    seed = int(tcfg.get("seed", 42))
    shuffle_files = bool(tcfg.get("patch_shuffle_files", True))
    shuffle_within_file = bool(tcfg.get("patch_shuffle_within_file", True))
    device = args.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    ds = PatchPTDataset(root, split)
    workers = [int(x.strip()) for x in args.workers.split(",") if x.strip()]
    print(f"[bench] split={split} rows={len(ds)} batch_size={batch_size} device={device}")
    print(f"[bench] workers={workers} warmup={args.warmup} steps={args.steps}")

    for nw in workers:
        bps, sps = run_once(
            ds=ds,
            batch_size=batch_size,
            num_workers=nw,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            seed=seed,
            shuffle_files=shuffle_files,
            shuffle_within_file=shuffle_within_file,
            warmup=args.warmup,
            steps=args.steps,
            device=device,
        )
        print(f"[bench] num_workers={nw:>2} batches/s={bps:.3f} samples/s={sps:.1f}")


if __name__ == "__main__":
    main()


from __future__ import annotations

from pathlib import Path

import torch


def _build_state(model, optimizer, scheduler, epoch: int, best_val: float) -> dict:
    state = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    return state


def save_checkpoint(checkpoint_dir: str, model, optimizer, scheduler, epoch: int, best_val: float, is_best: bool = False) -> None:
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    state = _build_state(model, optimizer, scheduler, epoch, best_val)
    torch.save(state, d / "current.pt")
    if is_best:
        torch.save(state, d / "best.pt")


def load_checkpoint(checkpoint_dir: str, model, optimizer=None, scheduler=None):
    d = Path(checkpoint_dir)
    current = d / "current.pt"
    if current.exists():
        state = torch.load(current, map_location="cpu")
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        return int(state["epoch"]) + 1, float(state["best_val"])

    # Backward compatibility with legacy split files.
    if not (d / "model.pt").exists():
        return 0, float("inf")
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    if optimizer is not None and (d / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(d / "optimizer.pt", map_location="cpu"))
    if scheduler is not None and (d / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(d / "scheduler.pt", map_location="cpu"))
    if (d / "trainer_state.pt").exists():
        state = torch.load(d / "trainer_state.pt", map_location="cpu")
        return int(state["epoch"]) + 1, float(state["best_val"])
    return 0, float("inf")

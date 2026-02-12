from __future__ import annotations

import torch
from tqdm import tqdm


class CUDAPrefetcher:
    def __init__(self, loader, device: str, prefetch_batches: int = 1):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=torch.device(device))
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.queue = []
        self._preload()

    def _to_device(self, b):
        return {
            "x": b["x"].to(self.device, non_blocking=True),
            "y": b["y"].to(self.device, non_blocking=True),
            "time7": b["time7"].to(self.device, non_blocking=True),
            "loc": b["loc"].to(self.device, non_blocking=True),
            "landcover_onehot": b["landcover_onehot"].to(self.device, non_blocking=True),
        }

    def _preload(self):
        while len(self.queue) < self.prefetch_batches:
            try:
                b = next(self.loader)
            except StopIteration:
                break
            with torch.cuda.stream(self.stream):
                self.queue.append(self._to_device(b))

    def next(self):
        torch.cuda.current_stream(device=torch.device(self.device)).wait_stream(self.stream)
        if not self.queue:
            return None
        b = self.queue.pop(0)
        self._preload()
        return b


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    train: bool,
    epoch: int,
    epochs: int,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    prefetch_batches: int = 0,
) -> tuple[float, float]:
    model.train() if train else model.eval()
    total = 0.0
    sse = 0.0
    n = 0
    mode = "train" if train else "val"
    pbar = tqdm(total=len(loader), desc=f"{mode} {epoch+1}/{epochs}", unit="batch")
    use_cuda_prefetch = str(device).startswith("cuda") and int(prefetch_batches) > 0
    prefetcher = CUDAPrefetcher(loader, device, prefetch_batches=prefetch_batches) if use_cuda_prefetch else None
    cpu_iter = iter(loader) if prefetcher is None else None
    for _ in range(len(loader)):
        if prefetcher is not None:
            b = prefetcher.next()
            if b is None:
                break
            x = b["x"]
            y = b["y"]
            t = b["time7"]
            loc = b["loc"]
            lc = b["landcover_onehot"]
        else:
            b = next(cpu_iter)
            x = b["x"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            t = b["time7"].to(device, non_blocking=True)
            loc = b["loc"].to(device, non_blocking=True)
            lc = b["landcover_onehot"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(use_amp and str(device).startswith("cuda"))):
                pred = model(x, t, loc, lc)
                loss = criterion(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler is not None and str(device).startswith("cuda"):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total += float(loss.item()) * x.size(0)
        sse += float(torch.sum((pred.detach() - y) ** 2).item())
        n += x.size(0)
        avg_loss = total / max(1, n)
        rmse = (sse / max(1, n)) ** 0.5
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", rmse=f"{rmse:.4f}")
        pbar.update(1)
    pbar.close()
    avg_loss = total / max(1, n)
    rmse = (sse / max(1, n)) ** 0.5
    return avg_loss, rmse

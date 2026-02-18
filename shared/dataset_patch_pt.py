from __future__ import annotations

import bisect
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchPTDataset(Dataset):
    def __init__(self, root: str, split: str, cache_size: int = 8):
        self.files = sorted(Path(root).glob(f'{split}_chunk_*.pt'))
        if not self.files:
            raise RuntimeError(f'No files for split={split} in {root}')
        self.cache_size = max(1, int(cache_size))
        self.counts = self._counts_exact()
        self.cum = np.cumsum(self.counts, dtype=np.int64)
        self.total = int(self.cum[-1])

    def _counts_exact(self) -> np.ndarray:
        counts = np.zeros((len(self.files),), dtype=np.int64)
        for i, fp in enumerate(self.files):
            try:
                d = torch.load(fp, map_location='cpu', mmap=True)
            except TypeError:
                d = torch.load(fp, map_location='cpu')
            counts[i] = int(d['x'].shape[0])
        return counts

    def __len__(self) -> int:
        return self.total

    def _rebuild_cum(self):
        self.cum = np.cumsum(self.counts, dtype=np.int64)
        self.total = int(self.cum[-1]) if len(self.cum) > 0 else 0

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_file(path: str) -> dict:
        try:
            d = torch.load(path, map_location='cpu', mmap=True)
        except TypeError:
            d = torch.load(path, map_location='cpu')
        return {
            'x': d['x'],
            'y': d['y'],
            'time7': d['time7'],
            'loc': d['loc'],
            'landcover_onehot': d['landcover_onehot'],
        }

    def _load_file_cached(self, path: str) -> dict:
        # cap cache growth in long runs by periodic global cache clear
        if self._load_file.cache_info().currsize > self.cache_size * 4:
            self._load_file.cache_clear()
        return self._load_file(path)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(f'Index out of range: idx={idx}, total={self.total}')

        # Keep this resilient even if chunk files changed after dataset init.
        for _ in range(3):
            file_i = bisect.bisect_right(self.cum, idx)
            if file_i >= len(self.files):
                file_i = len(self.files) - 1
            prev = 0 if file_i == 0 else int(self.cum[file_i - 1])
            j = int(idx - prev)

            d = self._load_file_cached(str(self.files[file_i]))
            n = int(d['x'].shape[0])

            # If real length differs from cached counts, refresh cumulative index.
            if n != int(self.counts[file_i]):
                self.counts[file_i] = n
                self._rebuild_cum()
                if idx >= self.total:
                    idx = self.total - 1
                continue

            if 0 <= j < n:
                return {
                    'x': d['x'][j],
                    'y': d['y'][j],
                    'time7': d['time7'][j],
                    'loc': d['loc'][j],
                    'landcover_onehot': d['landcover_onehot'][j],
                }

            # fallback: clamp local index in current file
            if n > 0:
                jj = max(0, min(j, n - 1))
                return {
                    'x': d['x'][jj],
                    'y': d['y'][jj],
                    'time7': d['time7'][jj],
                    'loc': d['loc'][jj],
                    'landcover_onehot': d['landcover_onehot'][jj],
                }

        raise IndexError(f'Failed to map global idx={idx} to any chunk')

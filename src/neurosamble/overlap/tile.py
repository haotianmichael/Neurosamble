"""
Read-side window tiling for the all-vs-all overlap pipeline (Neurosamble).

Extracted from ESA's ``evaluate/overlap_index.py`` -- only ``tile_read`` is
carried over here so the overlap pipeline does not drag in the FAISS-store build
helpers (and their heavy deps) that lived alongside it.

``tile_read`` slides a fixed-length window over the FULL recorded read. Same
convention as Rawsamble's all-vs-all (ava) seeding: we tile the recorded signal
and never add reverse-complement windows (a read and its RC are not both
recorded, so mixing them would only manufacture false, unchainable anchors).

Numpy-only: no torch / faiss / model imports.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def tile_read(signal, win: int = 2000, stride: int = 1000) -> List[Tuple[int, np.ndarray]]:
    """Slide a fixed-length window over the FULL read.

    Unlike the read->reference mapping path (which truncates each read to the
    first ``input_signal_len`` samples), here we tile the ENTIRE read so an
    overlap anywhere along two reads can surface anchors.

    Returns a list of ``(offset_samples, window_signal)``. The last partial
    window is kept (zero-padded downstream by ``preprocess_window``) iff its
    length ``>= win // 2``, otherwise it is dropped.
    """
    signal = np.asarray(signal, dtype=np.float32)
    n = int(signal.shape[0])
    win = int(win)
    stride = int(stride)
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")

    windows: List[Tuple[int, np.ndarray]] = []
    if n <= 0:
        return windows

    off = 0
    while off < n:
        w = signal[off : off + win]
        if w.shape[0] == win:
            windows.append((off, w))
            off += stride
        else:
            # Trailing partial window: keep only if long enough to be meaningful.
            if w.shape[0] >= win // 2:
                windows.append((off, w))
            break
    return windows
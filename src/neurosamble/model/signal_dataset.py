"""
Siamese positive-pair dataset for signal-domain contrastive training.

A positive pair is::

    x_1 = { 'signal': query_real_signal_window,   'attention_mask': mask_T }
    x_2 = { 'signal': ref_expected_signal_window, 'attention_mask': mask_T }

where the reference window ``x_2`` is produced by
``PoreModel.sequence_to_signal(reference[coord:coord+win_bp])`` and ``coord`` is
the ground-truth genomic start of the query read. Both sides go through the same
``SignalEncoder``; InfoNCE then pulls the two domains (noisy recorded vs. clean
expected) into one space.

Windows are fixed length (``input_signal_len`` samples), so the down-sampled
attention mask is well defined and aligns with the encoder output T; shorter
signals are zero-padded and the mask marks the valid region.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    """Reverse complement (same convention as upsert_signal.revcomp)."""
    return s.translate(_COMP)[::-1]


# --------------------------------------------------------------------------- #
# Shared preprocessing (also used at inference time)
# --------------------------------------------------------------------------- #
def znormalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    mean = float(signal.mean())
    std = float(signal.std())
    if std < 1e-6:
        std = 1e-6
    return (signal - mean) / std


def fix_length(signal: np.ndarray, target_len: int) -> Tuple[np.ndarray, int]:
    """Truncate or zero-pad ``signal`` to ``target_len``; return (array, valid_len)."""
    signal = np.asarray(signal, dtype=np.float32)
    n = int(signal.shape[0])
    if n >= target_len:
        return signal[:target_len].copy(), target_len
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = signal
    return out, n


def make_mask_T(valid_len: int, input_signal_len: int, downsample_factor: int) -> np.ndarray:
    """Build a T-resolution mask for a window with ``valid_len`` valid samples."""
    T = input_signal_len // downsample_factor
    valid_T = min(T, max(1, valid_len // downsample_factor))
    mask = np.zeros(T, dtype=np.int64)
    mask[:valid_T] = 1
    return mask


def preprocess_window(
    signal: np.ndarray, input_signal_len: int, downsample_factor: int
) -> Tuple[np.ndarray, np.ndarray]:
    """z-normalize -> fix length -> (signal[input_signal_len], mask[T])."""
    sig, valid = fix_length(znormalize(signal), input_signal_len)
    mask = make_mask_T(valid, input_signal_len, downsample_factor)
    return sig, mask


# --------------------------------------------------------------------------- #
class SignalPairDataset(IterableDataset):
    def __init__(
        self,
        query_signals: List[np.ndarray],
        query_coords: List[int],
        reference_seq: str,
        pore_model,
        unit_length: int,
        input_signal_len: int = 2000,
        downsample_factor: int = 5,
        samples_per_kmer: int = 9,
        hard_negatives: int = 0,
        hard_neg_min_bp: int = 30,
        hard_neg_max_bp: int = 300,
        query_strands: Optional[List[str]] = None,
        query_ends: Optional[List[int]] = None,
    ):
        super().__init__()
        assert len(query_signals) == len(query_coords)
        self.query_signals = query_signals
        self.query_coords = query_coords
        # Strand-aware positive pairs. A '-' read's recorded signal is the
        # revcomp of the reference substring, and (because the query is truncated
        # to input_signal_len ~ win_bp) its 5' end anchors at ``end - win_bp``,
        # NOT at ``start``. Without this, ~half the pairs (all reverse reads) get
        # a positive that is a DIFFERENT stretch of DNA — corrupting training.
        self.query_strands = query_strands
        self.query_ends = query_ends
        if query_strands is not None:
            assert len(query_strands) == len(query_signals)
        if query_ends is not None:
            assert len(query_ends) == len(query_signals)
        self.reference_seq = reference_seq
        self.pore_model = pore_model
        self.input_signal_len = input_signal_len
        self.downsample_factor = downsample_factor
        # The training reference span MUST equal the index window span
        # (``unit_length``) — otherwise the encoder learns to match a reference
        # vector length that does not exist in the FAISS index, and the true
        # window is pushed out of the top-k (recall can drop below random).
        self.win_bp = unit_length
        self.max_start = max(0, len(reference_seq) - self.win_bp)

        self.hard_negatives = hard_negatives
        self.hard_neg_min_bp = hard_neg_min_bp
        self.hard_neg_max_bp = hard_neg_max_bp

    def _ref_window(self, coord: int, strand: str = "+"):
        """Expected-signal window (z-normed, fixed length) at a genomic coord.

        ``strand='-'`` reverse-complements the reference bases first, matching
        the revcomp windows the FAISS index stores and the physical DNA a
        reverse-strand read traverses.
        """
        coord = int(min(max(coord, 0), self.max_start))
        ref_bases = self.reference_seq[coord : coord + self.win_bp]
        if strand == "-":
            ref_bases = revcomp(ref_bases)
        ref_signal = self.pore_model.sequence_to_signal(ref_bases)
        return preprocess_window(ref_signal, self.input_signal_len, self.downsample_factor)

    def _anchor(self, idx: int) -> Tuple[int, str]:
        """Genomic anchor + strand of the query's 5' end (what the truncated
        query signal actually starts at)."""
        strand = self.query_strands[idx] if self.query_strands is not None else "+"
        end = self.query_ends[idx] if self.query_ends is not None else None
        if strand == "-" and end is not None:
            return max(0, int(end) - self.win_bp), strand   # reverse read's 5' end
        return int(self.query_coords[idx]), strand

    def _pair(self, idx: int):
        anchor, strand = self._anchor(idx)

        q_sig, q_mask = preprocess_window(
            self.query_signals[idx], self.input_signal_len, self.downsample_factor
        )
        r_sig, r_mask = self._ref_window(anchor, strand)

        x_1 = {"signal": q_sig, "attention_mask": q_mask}
        x_2 = {"signal": r_sig, "attention_mask": r_mask}

        if self.hard_negatives <= 0:
            return x_1, x_2

        # Near-coordinate hard negatives: same strand & span, offset by [min,max]
        # bp on a random side. These force fine positional discrimination that
        # in-batch random negatives (almost always far away) never exercise. The
        # strand MUST match the anchor, else they collapse into trivial negatives.
        neg_sigs, neg_masks = [], []
        for _ in range(self.hard_negatives):
            sign = 1 if np.random.rand() < 0.5 else -1
            offset = np.random.randint(self.hard_neg_min_bp, self.hard_neg_max_bp + 1)
            ns, nm = self._ref_window(anchor + sign * offset, strand)
            neg_sigs.append(ns)
            neg_masks.append(nm)
        x_neg = {
            "signal": np.stack(neg_sigs),           # [H, L]
            "attention_mask": np.stack(neg_masks),  # [H, T]
        }
        return x_1, x_2, x_neg

    def __iter__(self):
        n = len(self.query_signals)
        while True:
            idx = int(torch.randint(0, n, (1,)).item())
            yield self._pair(idx)


def signal_collate(batch):
    """Stack (x_1, x_2) — or (x_1, x_2, x_neg) when hard negatives are on.

    Without negatives returns ``(x_1, x_2)`` (the shape ``ContrastiveTrainer``
    expects, so the H=0 path stays on the unmodified trainer). With negatives
    returns ``(x_1, x_2, x_neg)`` where x_neg is ``[B, H, L]`` / ``[B, H, T]``.
    """
    has_neg = len(batch[0]) == 3

    def stack(dicts):
        signals = torch.from_numpy(np.stack([d["signal"] for d in dicts])).float()
        masks = torch.from_numpy(np.stack([d["attention_mask"] for d in dicts])).long()
        return {"signal": signals, "attention_mask": masks}

    x_1 = stack([b[0] for b in batch])
    x_2 = stack([b[1] for b in batch])
    if not has_neg:
        return x_1, x_2

    neg_signal = torch.from_numpy(np.stack([b[2]["signal"] for b in batch])).float()
    neg_mask = torch.from_numpy(np.stack([b[2]["attention_mask"] for b in batch])).long()
    x_neg = {"signal": neg_signal, "attention_mask": neg_mask}
    return x_1, x_2, x_neg

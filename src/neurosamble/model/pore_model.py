"""
ONT k-mer pore model: convert a base sequence into its *expected* current signal.

This mirrors the RawHash reference-side trick: instead of recording signal for
the reference, we translate reference bases into the piecewise-constant current
level that each k-mer is expected to produce, using an ONT k-mer pore model
table (R9 = 6-mer, R10 = 9-mer). Every k-mer contributes one current level; we
hold it for a fixed number of samples (no dwell-time variation) so the mapping
is deterministic.

The real R9.4 6-mer table can be taken from CMU-SAFARI/RawHash (or ONT
remora/bonito). Point ``kmer_table_path`` at it (whitespace-delimited
``<kmer> <level_pA> [<stdv>]`` per line). If no table is supplied a deterministic
*synthetic* model is used so the plumbing is testable end-to-end — this is a
smoke-test aid only and is NOT valid for a real go/no-go measurement.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional

import numpy as np

_BASES = "ACGT"


class PoreModel:
    def __init__(
        self,
        kmer_table_path: Optional[str] = None,
        kmer_len: int = 6,
        samples_per_kmer: int = 9,
    ):
        """
        Args:
            kmer_table_path: Path to an ONT k-mer -> current-level table. If None,
                falls back to ``$PORE_MODEL_PATH`` and then to a synthetic model.
            kmer_len: k-mer length (6 for R9, 9 for R10).
            samples_per_kmer: fixed samples emitted per k-mer (dwell time).
        """
        self.kmer_len = kmer_len
        self.samples_per_kmer = samples_per_kmer

        if kmer_table_path is None:
            kmer_table_path = os.environ.get("PORE_MODEL_PATH")

        self.synthetic = kmer_table_path is None
        if self.synthetic:
            self.table: Dict[str, float] = {}
            self._default_level = 0.0
            self._level_mean = 90.0   # rough pA scale, only used for synthetic
            self._level_std = 20.0
        else:
            self.table = self._load_table(kmer_table_path)
            levels = np.fromiter(self.table.values(), dtype=np.float32)
            self._level_mean = float(levels.mean())
            self._level_std = float(levels.std() + 1e-6)
            self._default_level = self._level_mean

    # ------------------------------------------------------------------ #
    def _load_table(self, path: str) -> Dict[str, float]:
        table: Dict[str, float] = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                kmer = parts[0].upper()
                # skip a possible header row like "kmer level_mean ..."
                if not set(kmer) <= set(_BASES):
                    continue
                if len(kmer) != self.kmer_len:
                    self.kmer_len = len(kmer)
                table[kmer] = float(parts[1])
        if not table:
            raise ValueError(f"No valid k-mer rows parsed from pore model: {path}")
        return table

    def _synthetic_level(self, kmer: str) -> float:
        # Deterministic pseudo-level in a plausible pA range from a hash of the kmer.
        h = int(hashlib.md5(kmer.encode()).hexdigest()[:8], 16)
        unit = (h % 100000) / 100000.0  # [0, 1)
        return self._level_mean + (unit - 0.5) * 2.0 * self._level_std

    def _level(self, kmer: str) -> float:
        if self.synthetic:
            return self._synthetic_level(kmer)
        return self.table.get(kmer, self._default_level)

    # ------------------------------------------------------------------ #
    def sequence_to_signal(self, seq: str) -> np.ndarray:
        """Base string -> piecewise-constant expected signal (1-D float32)."""
        seq = seq.upper()
        k = self.kmer_len
        if len(seq) < k:
            return np.zeros(0, dtype=np.float32)

        n_kmers = len(seq) - k + 1
        levels = np.empty(n_kmers, dtype=np.float32)
        for i in range(n_kmers):
            levels[i] = self._level(seq[i : i + k])

        signal = np.repeat(levels, self.samples_per_kmer)
        return signal.astype(np.float32)

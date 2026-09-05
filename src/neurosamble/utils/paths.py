"""
Central path & data-location config for Neurosamble.

Single source of truth for every filesystem path. Override any of these without
editing code via environment variables, e.g.:

    export NEUROSAMBLE_DATA_DIR=/some/other/ecoli
    export PORE_MODEL_PATH=/some/other/table.model

Import in scripts:

    from neurosamble.utils.paths import P
    print(P.blow5, P.pairs_pkl, P.pore_model)
"""
from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


class P:
    # --- repo layout (derived; don't hardcode) ---------------------------- #
    # this file lives at .../Neurosamble/src/neurosamble/utils/paths.py
    SRC        = Path(__file__).resolve().parents[2]     # .../Neurosamble/src
    REPO       = SRC.parent                              # .../Neurosamble
    OUT_DIR    = _env("NEUROSAMBLE_OUT_DIR", str(REPO / "data"))

    # --- ecoli R9 dataset: vendored into repo data/ ----------------------- #
    DATA_DIR   = _env("NEUROSAMBLE_DATA_DIR", str(REPO / "data"))
    blow5      = DATA_DIR / "ecoli_R9.blow5"
    paf        = DATA_DIR / "truth.paf"
    ref        = DATA_DIR / "ref.fa"

    # --- ONT pore model table: vendored into repo data/ ------------------- #
    pore_model = _env("PORE_MODEL_PATH", str(REPO / "data" / "pore_r9.4_6mer.model"))

    # --- generated artifacts (in this repo's data/) ----------------------- #
    pairs_pkl        = Path(OUT_DIR) / "ecoli_pairs.pkl"
    baseline_encoder = Path(OUT_DIR) / "baseline_encoder.pt"

    @classmethod
    def ensure_out(cls) -> None:
        Path(cls.OUT_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def check_inputs(cls) -> None:
        """Fail early with a clear message if a required input is missing."""
        missing = [str(p) for p in (cls.blow5, cls.paf, cls.ref, Path(cls.pore_model))
                   if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required input(s):\n  " + "\n  ".join(missing) +
                "\n(edit neurosamble/utils/paths.py or set the matching env var)"
            )
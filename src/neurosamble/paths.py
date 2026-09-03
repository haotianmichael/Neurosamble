"""
Central path & data-location config for Neurosamble.

Single source of truth for every filesystem path used by the harness scripts.
Override any of these without editing code via environment variables, e.g.:

    export NEUROSAMBLE_DATA_DIR=/some/other/ecoli
    export PORE_MODEL_PATH=/some/other/table.model

Import in scripts:

    from neurosamble.paths import P
    print(P.blow5, P.pairs_pkl, P.pore_model)
"""
from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


class P:
    # --- repo layout (derived; don't hardcode) ---------------------------- #
    # paths.py lives at .../Neurosamble/src/neurosamble/paths.py
    SRC        = Path(__file__).resolve().parents[1]      # .../Neurosamble/src
    REPO       = SRC.parent                               # .../Neurosamble
    OUT_DIR    = _env("NEUROSAMBLE_OUT_DIR", str(REPO / "data"))

    # --- raw ecoli R9 dataset (external, in CALL_ESA) --------------------- #
    DATA_DIR   = _env("NEUROSAMBLE_DATA_DIR",
                      "/home/nfs/mahaotian/ESA/CALL_ESA/data/d2_ecoli_r94")
    blow5      = DATA_DIR / "ecoli_R9.blow5"
    paf        = DATA_DIR / "truth.paf"
    ref        = DATA_DIR / "ref.fa"

    # --- ONT pore model table (external, in Rawhash2) --------------------- #
    pore_model = _env(
        "PORE_MODEL_PATH",
        "/home/nfs/mahaotian/ESA/Rawhash2/extern/kmer_models/legacy/"
        "legacy_r9.4_180mv_450bps_6mer/template_median68pA.model",
    )

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
                "\n(edit neurosamble/paths.py or set the matching env var)"
            )
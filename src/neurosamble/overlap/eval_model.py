"""
Self-contained signal-domain inference model for the overlap pipeline.

This replaces ESA's ``inference_signal.SignalEvalModel`` + ``pilot_recall.load_encoder``.
Loads the checkpoint that ``harness/finetune.py`` writes and builds the model
from its plain ``signal_config`` dict.

``SignalEvalModel.encode(list_of_signals) -> np.ndarray[N, D]`` runs the SAME
preprocessing as training (z-normalize + fixed-length window + T-resolution
mask), pushes windows through the frozen ``SignalEncoder`` + ``AveragePooler``,
L2-normalizes per vector, and returns embeddings.

PERF NOTES (Neurosamble encode acceleration):
  * Preprocessing (z-norm + masks) is VECTORIZED on the GPU -- the old per-window
    numpy loop is gone; we build one [B, L] batch, z-normalize per row over the
    valid region on-device, and derive the T-resolution mask with two arange
    compares. Numerically equivalent to ``preprocess_window`` (ddof=0, std floor
    1e-6, zero padding after z-norm).
  * The encoder forward runs under ``torch.autocast`` (fp16 by default): conv +
    Mamba/attention matmuls run in fp16 (LayerNorm/pooling stay fp32), which both
    (fp16 not bf16: bf16 PTX needs sm_80+, and this toolchain targets sm_70 for the
    Mamba2 Triton kernels; fp16 compiles there)
    speeds up the forward and roughly halves activation memory -- so the GPU
    batch can be pushed up ~10x from the old 256.
  * ``encode`` accepts ``out_dtype`` so the caller can persist fp16 shards
    (halves the embedding files + the query-side memmap / index RAM).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from neurosamble.model.signal_encoder import SignalEncoder
from neurosamble.model.pooling import AveragePooler


def _as_last_hidden(out):
    """Return the ``[B, T, D]`` hidden states from an encoder forward."""
    if isinstance(out, dict):
        for key in ("last_hidden", "last_hidden_state", "hidden_states", "logits"):
            if key in out:
                return out[key]
        raise KeyError(f"encoder returned a dict without a known hidden-state key: {list(out)}")
    return out


class SignalEvalModel:
    """Mirror of ESA's ``inference_signal.SignalEvalModel`` (self-contained, GPU-batched)."""

    def __init__(
        self,
        encoder,
        pooling,
        device,
        input_signal_len: int = 2000,
        downsample_factor: int = 5,
        embedding_dim: int = 384,
        batch_size: int = 2048,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.encoder = encoder.to(device)
        self.pooling = pooling.to(device)
        self.device = device
        self.input_signal_len = input_signal_len
        self.downsample_factor = downsample_factor
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self._is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.encoder.eval()

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    # ------------------------------------------------------------------ #
    def _prep_batch(self, signals: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized z-norm + T-mask on the GPU.

        Equivalent to ``preprocess_window`` applied per window, but batched:
          * z-normalize each row over its VALID region (mean/std, ddof=0, std
            floored at 1e-6), then zero the padding -- matches ``znormalize``
            then ``fix_length`` (which pads with zeros AFTER z-norm).
          * T-resolution mask: valid_T = clamp(valid // ds, 1, T).

        ``tile_read`` never emits a window longer than ``input_signal_len``, so
        the only length adjustment here is zero-padding of the trailing partial.
        """
        L = self.input_signal_len
        ds = self.downsample_factor
        T = L // ds
        B = len(signals)

        sig = np.zeros((B, L), dtype=np.float32)
        valid = np.empty(B, dtype=np.int64)
        for i, s in enumerate(signals):
            s = np.asarray(s, dtype=np.float32)
            n = int(s.shape[0])
            if n > L:
                n = L
                sig[i] = s[:L]
            elif n > 0:
                sig[i, :n] = s
            valid[i] = n

        sig_t = torch.from_numpy(sig).to(self.device, non_blocking=True)
        valid_t = torch.from_numpy(valid).to(self.device, non_blocking=True)

        ar = torch.arange(L, device=self.device)
        m = ar.unsqueeze(0) < valid_t.unsqueeze(1)                 # [B, L] bool
        denom = valid_t.clamp(min=1).unsqueeze(1).to(sig_t.dtype)  # avoid /0
        mean = sig_t.sum(dim=1, keepdim=True) / denom              # padding is 0
        diff = (sig_t - mean) * m                                  # zero padding
        var = (diff * diff).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var).clamp(min=1e-6)
        sig_norm = diff / std                                      # padding stays 0

        arT = torch.arange(T, device=self.device)
        vt = (valid_t // ds).clamp(min=1, max=T)
        mask_T = (arT.unsqueeze(0) < vt.unsqueeze(1)).long()       # [B, T]
        return sig_norm, mask_T

    def encode(self, signals: List[np.ndarray], out_dtype=np.float32) -> np.ndarray:
        outputs = []
        bs = self.batch_size
        amp_device = "cuda" if self._is_cuda else "cpu"
        with torch.no_grad():
            self.encoder.eval()
            for start in range(0, len(signals), bs):
                chunk = signals[start : start + bs]
                signal, attention_mask = self._prep_batch(chunk)
                with torch.autocast(device_type=amp_device, dtype=self.amp_dtype,
                                    enabled=(self._is_cuda and self.use_amp)):
                    out = self.encoder(signal=signal, attention_mask=attention_mask)
                    last_hidden = _as_last_hidden(out)
                # pool + normalize in fp32 for stability regardless of AMP
                y = self.pooling(last_hidden.float(), attention_mask=attention_mask)
                y = F.normalize(y, dim=-1)
                outputs.append(y.to(torch.float32).cpu().numpy())
        if not outputs:
            return np.zeros((0, self.embedding_dim), dtype=out_dtype)
        return np.concatenate(outputs, axis=0).astype(out_dtype)


def load_encoder(path, device, batch_size: int = 2048, use_amp: bool = True):
    """Load the frozen encoder checkpoint written by ``harness/finetune.py``.

    Returns ``(SignalEvalModel, signal_config_dict)``.
    """
    ckpt = torch.load(path, map_location="cpu")
    signal_config = dict(ckpt["signal_config"])

    encoder = SignalEncoder(**signal_config)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()
    pooling = AveragePooler()

    model = SignalEvalModel(
        encoder=encoder,
        pooling=pooling,
        device=device,
        input_signal_len=int(signal_config.get("input_signal_len", 2000)),
        downsample_factor=int(signal_config.get("downsample_factor", 5)),
        embedding_dim=int(signal_config.get("embedding_dim", 384)),
        batch_size=batch_size,
        use_amp=use_amp,
    )
    amp_str = "off" if not use_amp else ("fp16" if model.amp_dtype == torch.float16 else "bf16")
    print(f"[info] loaded encoder <- {path} (D={model.get_sentence_embedding_dimension()}, "
          f"batch_size={batch_size}, amp={amp_str})", flush=True)
    return model, signal_config
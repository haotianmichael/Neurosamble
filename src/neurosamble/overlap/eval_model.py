"""
Self-contained signal-domain inference model for the overlap pipeline.

This replaces ESA's ``inference_signal.SignalEvalModel`` + ``pilot_recall.load_encoder``,
which depended on a pydantic ``SignalModelConfigSchema`` and a
``signal_encoder_from_config`` helper that Neurosamble does not ship. Here we load
the checkpoint that ``harness/finetune.py`` writes directly and build the model
from its plain ``signal_config`` dict.

``SignalEvalModel.encode(list_of_signals) -> np.ndarray[N, D]`` runs the SAME
preprocessing as training (z-normalize + fixed-length window + T-resolution
mask), pushes windows through the frozen ``SignalEncoder`` + ``AveragePooler``,
L2-normalizes per vector, and returns float32 embeddings. It exposes
``get_sentence_embedding_dimension`` + ``encode`` so it drops into the index /
query steps unchanged.

The model layer is used read-only: nothing under ``neurosamble.model`` is
modified, only imported and called.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from neurosamble.model.signal_encoder import SignalEncoder
from neurosamble.model.pooling import AveragePooler
from neurosamble.model.signal_dataset import preprocess_window


def _as_last_hidden(out):
    """Return the ``[B, T, D]`` hidden states from an encoder forward.

    Neurosamble's ``SignalEncoder.forward`` returns a plain tensor, but we handle
    a dict return (e.g. HuggingFace-style ``last_hidden_state``) defensively so a
    future encoder change does not silently break encoding.
    """
    if isinstance(out, dict):
        for key in ("last_hidden", "last_hidden_state", "hidden_states", "logits"):
            if key in out:
                return out[key]
        raise KeyError(f"encoder returned a dict without a known hidden-state key: {list(out)}")
    return out


class SignalEvalModel:
    """Mirror of ESA's ``inference_signal.SignalEvalModel`` (self-contained)."""

    def __init__(
        self,
        encoder,
        pooling,
        device,
        input_signal_len: int = 2000,
        downsample_factor: int = 5,
        embedding_dim: int = 384,
        batch_size: int = 256,
    ):
        self.encoder = encoder.to(device)
        self.pooling = pooling.to(device)
        self.device = device
        self.input_signal_len = input_signal_len
        self.downsample_factor = downsample_factor
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.encoder.eval()

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def _prep_batch(self, signals: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        sigs, masks = [], []
        for s in signals:
            sig, mask = preprocess_window(s, self.input_signal_len, self.downsample_factor)
            sigs.append(sig)
            masks.append(mask)
        signal = torch.from_numpy(np.stack(sigs)).float().to(self.device)
        attention_mask = torch.from_numpy(np.stack(masks)).long().to(self.device)
        return signal, attention_mask

    def encode(self, signals: List[np.ndarray]) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            self.encoder.eval()
            for start in range(0, len(signals), self.batch_size):
                chunk = signals[start : start + self.batch_size]
                signal, attention_mask = self._prep_batch(chunk)
                out = self.encoder(signal=signal, attention_mask=attention_mask)
                last_hidden = _as_last_hidden(out)
                y = self.pooling(last_hidden, attention_mask=attention_mask)
                y = F.normalize(y, dim=-1)
                outputs.append(y.detach().cpu().numpy())
        if not outputs:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32)


def load_encoder(path, device):
    """Load the frozen encoder checkpoint written by ``harness/finetune.py``.

    The checkpoint is ``{"model": encoder.state_dict(), "signal_config": {...}}``
    where ``signal_config`` holds exactly the ``SignalEncoder.__init__`` kwargs
    (encoder_type, conv_channels_1, conv_channels_2, conv_kernel_1, n_blocks,
    num_heads, embedding_dim, dropout, input_signal_len, downsample_factor).

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
    )
    print(f"[info] loaded encoder <- {path} (D={model.get_sentence_embedding_dimension()})", flush=True)
    return model, signal_config
"""Custom CUDA-kernel backend for trainable Linear layers.

Public API::

    from neurosamble.kernels import MyLinear, CustomLinearFn

``MyLinear`` is a drop-in ``nn.Linear`` replacement with a ``use_custom``
switch; ``CustomLinearFn`` is the underlying autograd Function that routes all
three Linear GEMMs through the (JIT-compiled, replaceable) kernel in
``csrc/gemm.cu``.
"""
from .layers import MyLinear
from .ops import CustomLinearFn

__all__ = ["MyLinear", "CustomLinearFn"]
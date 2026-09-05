"""``MyLinear`` -- a drop-in replacement for ``nn.Linear`` with a custom-kernel
switch.

* Parameter names, shapes, and initialization are identical to ``nn.Linear``,
  so ``MyLinear`` loads/saves state dicts interchangeably and, with
  ``use_custom=False``, is numerically identical to ``nn.Linear``.
* ``use_custom=True`` routes forward/backward through ``CustomLinearFn`` (the
  custom CUDA GEMM); ``use_custom=False`` falls back to ``F.linear``.
* ``from_linear`` copies weights from an existing ``nn.Linear`` so a call site
  can be swapped without reinitializing.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import CustomLinearFn


class MyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_custom: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_custom = use_custom

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Identical initialization to torch.nn.Linear.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_custom:
            return CustomLinearFn.apply(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, use_custom: bool = True) -> "MyLinear":
        """Build a ``MyLinear`` that reuses an existing ``nn.Linear``'s weights."""
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            use_custom=use_custom,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, use_custom={self.use_custom}"
        )

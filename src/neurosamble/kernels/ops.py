"""JIT-compiled custom GEMM backend + autograd wrapper for a trainable Linear.

The extension in ``csrc/hgemm_cutlass.cu`` exposes a single primitive,
``gemm(A, B) -> A @ B``.  All three GEMMs of an ``nn.Linear`` (forward + the two
backward passes) are composed from it here, inside a ``torch.autograd.Function``
so the custom kernel is fully trainable.

The extension is compiled lazily on first use with ``cpp_extension.load`` (JIT,
not AOT) and cached at module scope, so importing this module is cheap and does
not require a CUDA toolchain until a custom Linear is actually run.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# Module-level cache for the compiled extension (see ``_load``).
_EXT = None


def _load():
    """Compile (once) and return the custom GEMM extension module."""
    global _EXT
    if _EXT is not None:
        return _EXT

    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(this_dir, "csrc")
    # kernels/ -> neurosamble/ -> src/ -> <repo root>/third-party/cutlass/include
    cutlass_include = os.path.abspath(
        os.path.join(this_dir, "..", "..", "..", "third-party", "cutlass", "include")
    )
    cutlass_util_include = os.path.abspath(
        os.path.join(this_dir, "..", "..", "..", "third-party", "cutlass", "tools", "util", "include")
    )

    # Target the current device's compute capability (e.g. "8.0" / "12.0").
    # Only meaningful when a GPU is visible at compile time.
    if torch.cuda.is_available():
        os.environ.setdefault(
            "TORCH_CUDA_ARCH_LIST",
            ".".join(map(str, torch.cuda.get_device_capability())),
        )

    _EXT = load(
        name="neurosamble_gemm",
        sources=[os.path.join(csrc_dir, "hgemm_cutlass.cu")],
        extra_include_paths=[cutlass_include, cutlass_util_include],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-DNDEBUG",
        ],
        extra_cflags=["-std=c++17"],
        verbose=bool(int(os.environ.get("NEUROSAMBLE_KERNELS_VERBOSE", "0"))),
    )
    return _EXT


class CustomLinearFn(torch.autograd.Function):
    """Autograd wrapper for ``Y = X @ W^T (+ b)`` routed through the custom GEMM.

    ``weight`` follows the ``nn.Linear`` convention with shape ``[out, in]``.
    ``x`` may have any leading shape ``[*, in]``; it is flattened to 2-D for the
    GEMM and restored on the way out.

    The custom primitive is a plain matmul ``gemm(A, B) = A @ B`` (both 2-D,
    row-major), so each of the three Linear GEMMs is expressed in that form.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor]):
        ext = _load()

        orig_shape = x.shape
        in_features = orig_shape[-1]
        out_features = weight.size(0)

        x2d = x.reshape(-1, in_features).contiguous()
        wt = weight.t().contiguous()          # [in, out]
        y2d = ext.gemm(x2d, wt)               # x2d @ wt = X @ W^T -> [M, out]
        if bias is not None:
            y2d = y2d + bias

        ctx.save_for_backward(x2d, weight)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape

        return y2d.reshape(*orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ext = _load()
        x2d, weight = ctx.saved_tensors
        out_features = weight.size(0)

        gy2d = grad_output.reshape(-1, out_features).contiguous()

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # dX = dY @ W   ->  [M, out] @ [out, in] = [M, in]
            gx2d = ext.gemm(gy2d, weight)
            grad_input = gx2d.reshape(ctx.orig_shape)

        if ctx.needs_input_grad[1]:
            # dW = dY^T @ X  ->  [out, M] @ [M, in] = [out, in]
            grad_weight = ext.gemm(gy2d.t().contiguous(), x2d)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            # db = sum over the batch dimension
            grad_bias = gy2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias
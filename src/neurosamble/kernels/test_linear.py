"""Acceptance test: custom-kernel Linear vs. torch ``nn.Linear``.

Run with::

    PYTHONPATH=$PWD/src python -m neurosamble.kernels.test_linear

Builds a ``MyLinear`` (custom kernels ON) and an ``nn.Linear`` with identical
weights, then compares forward output and all three gradients
(``grad_input``, ``grad_weight``, ``grad_bias``) in both fp16 and fp32 via
relative error ``||custom - torch|| / ||torch||``.

Prints PASS iff every relative error is below tolerance (1e-2 for fp16, 1e-4
for fp32) and exits nonzero on failure. A GPU is required (the custom kernel is
CUDA-only).
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from neurosamble.kernels.layers import MyLinear

# (dtype, tolerance) pairs to exercise.
CASES = [
    (torch.float32, 1e-4),
    (torch.float16, 1e-2),
]

M, IN_FEATURES, OUT_FEATURES = 256, 512, 384


def relative_error(actual: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    diff = (actual - ref).float()
    return (torch.norm(diff) / (torch.norm(ref.float()) + eps)).item()


def run_case(dtype: torch.dtype, tol: float, device: torch.device) -> bool:
    torch.manual_seed(0)

    # Reference nn.Linear, then a MyLinear that copies its exact weights.
    ref = nn.Linear(IN_FEATURES, OUT_FEATURES)
    custom = MyLinear.from_linear(ref, use_custom=True)
    ref = ref.to(device=device, dtype=dtype)
    custom = custom.to(device=device, dtype=dtype)

    x = torch.randn(M, IN_FEATURES, device=device, dtype=dtype)
    x_ref = x.clone().detach().requires_grad_(True)
    x_cus = x.clone().detach().requires_grad_(True)

    y_ref = ref(x_ref)
    y_cus = custom(x_cus)
    re_fwd = relative_error(y_cus, y_ref)

    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)
    y_cus.backward(grad_out.clone())

    re_dx = relative_error(x_cus.grad, x_ref.grad)
    re_dw = relative_error(custom.weight.grad, ref.weight.grad)
    re_db = relative_error(custom.bias.grad, ref.bias.grad)

    results = {
        "forward": re_fwd,
        "grad_input": re_dx,
        "grad_weight": re_dw,
        "grad_bias": re_db,
    }
    ok = all(re < tol for re in results.values())

    print(f"[{str(dtype):<14}] tol={tol:g}")
    for name, re in results.items():
        flag = "ok " if re < tol else "BAD"
        print(f"    {flag} {name:<12} RE = {re:.3e}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available; the custom GEMM kernel is CUDA-only.")
        return 1

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}  "
          f"(capability {'.'.join(map(str, torch.cuda.get_device_capability()))})")
    print("Compiling extension on first run; this may take a minute...\n")

    all_ok = True
    for dtype, tol in CASES:
        all_ok &= run_case(dtype, tol, device)
        print()

    print("=" * 40)
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

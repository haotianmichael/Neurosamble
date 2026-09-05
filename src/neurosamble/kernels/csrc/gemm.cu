// ============================================================================
//  gemm.cu  --  minimal, correct reference GEMM for the custom-kernel backend
// ============================================================================
//
//  >>> REPLACE THIS with CuTe/CUTLASS kernels later. KEEP THE SAME FUNCTION
//  >>> SIGNATURES.  ``ops.py`` (the autograd.Function) depends ONLY on the
//  >>> pybind ABI exposed at the bottom of this file:
//  >>>
//  >>>     gemm(A, B) -> A @ B        # 2-D, row-major, CUDA, fp32 or fp16
//  >>>
//  >>> A Linear layer's three GEMMs all route through this single primitive
//  >>> (see ops.py::CustomLinearFn):
//  >>>
//  >>>     forward :  Y  = X  @ Wᵀ            gemm(X,  Wᵀ)
//  >>>     dX      = dY @ W                   gemm(dY, W)
//  >>>     dW      = dYᵀ @ X                  gemm(dYᵀ, X)
//  >>>
//  >>> So swapping the body of ``gemm`` (or adding transpose-fused variants
//  >>> with the same names) for hand-written CuTe/CUTLASS is enough to make
//  >>> the whole training path use custom kernels -- no Python changes needed.
//
//  This implementation is intentionally a NAIVE shared-memory tiled GEMM.
//  Correctness matters; speed does NOT.  Half inputs accumulate in fp32 so the
//  numerics match PyTorch's cuBLAS matmul closely enough for the acceptance
//  test in test_linear.py.
//
//  The CUTLASS include path (third-party/cutlass/include) is put on the
//  compiler search path by ops.py via ``extra_include_paths`` so that the
//  future CuTe/CUTLASS drop-in can ``#include <cute/...>`` here directly.
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16

// Naive tiled GEMM: C[M,N] = A[M,K] @ B[K,N], all row-major.
// Accumulation is always done in float, regardless of input dtype, to keep the
// fp16 path numerically close to PyTorch (which accumulates matmuls in fp32).
template <typename scalar_t>
__global__ void naive_tiled_gemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  const int num_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    const int b_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? static_cast<float>(A[row * K + a_col]) : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? static_cast<float>(B[b_row * N + col]) : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = static_cast<scalar_t>(acc);
  }
}

// gemm(A, B) -> A @ B.  Public ABI -- do not change the signature.
torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "gemm: A and B must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "gemm: A and B must be 2-D");
  TORCH_CHECK(A.size(1) == B.size(0),
              "gemm: inner dimensions must match (A[M,K] @ B[K,N])");
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "gemm: A and B must share the same dtype");

  auto Ac = A.contiguous();
  auto Bc = B.contiguous();

  const int M = static_cast<int>(Ac.size(0));
  const int K = static_cast<int>(Ac.size(1));
  const int N = static_cast<int>(Bc.size(1));

  auto C = torch::empty({M, N}, Ac.options());

  const dim3 block(TILE, TILE);
  const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(Ac.scalar_type(), "gemm", [&] {
    naive_tiled_gemm_kernel<scalar_t><<<grid, block>>>(
        Ac.data_ptr<scalar_t>(),
        Bc.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M, N, K);
  });
  C10_CUDA_CHECK(cudaGetLastError());

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm,
        "C = A @ B  (naive tiled reference GEMM; replace with CuTe/CUTLASS, "
        "keep the signature)");
}
#include <torch/extension.h>
#include <torch/types.h>
#include "hgemm_cute.cuh"


#define CHECK_TORCH_TENSOR_DTYPE(T, DTYPE)                                                                            \
  do {                                                                                                                \
    if ((T).options().dtype() != (DTYPE)) {                                                                           \
      std::cerr << "Tensor dtype mismatch! Expected: " << (DTYPE) << ", but got: " << (T).options().dtype() << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;                                                          \
      std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                 \
  } while (0);

#define CHECK_TORCH_TENSOR_SHAPE(T, M, N)                                                                             \
  do {                                                                                                                \
    auto actual_shape = (T).sizes();                                                                                  \
    if (actual_shape != torch::IntArrayRef({M, N})) {                                                                 \
      std::cerr << "Tensor shape mismatch! Expected: " << torch::IntArrayRef({M, N}) << ", but got: " << actual_shape \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                 \
  } while (0);

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                            \
  [&] {                                                                                                               \
    if (COND) {                                                                                                       \
      constexpr static bool CONST_NAME = true;                                                                        \
      return __VA_ARGS__();                                                                                           \
    } else {                                                                                                          \
      constexpr static bool CONST_NAME = false;                                                                       \
      return __VA_ARGS__();                                                                                           \
    }                                                                                                                 \
  }()

template <typename T> constexpr torch::ScalarType to_torch_scalar_type() {
  if constexpr (std::is_same_v<T, cute::half_t>)
    return torch::kHalf;
  else if constexpr (std::is_same_v<T, cute::bfloat16_t>)
    return torch::kBFloat16;
  else if constexpr (std::is_same_v<T, float>)
    return torch::kFloat32;
  else if constexpr (std::is_same_v<T, cute::float_e4m3_t>)
    return torch::kFloat8_e4m3fn;
  else if constexpr (std::is_same_v<T, cute::float_e5m2_t>)
    return torch::kFloat8_e5m2;
  else
    throw std::runtime_error("Unsupported type!");
}

template <typename ComputeTypeC, typename OutType> constexpr bool needs_precision_conversion() {
  return !std::is_same_v<ComputeTypeC, OutType>;
}

template <int kBlockM,
          int kBlockN,
          int kBlockK,
          int G2S_Stages,
          typename OutType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename ComputeTypeC = OutType>
torch::Tensor run_hgemm(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto M = a.size(0);
  auto N = b.size(0);
  auto K = a.size(1);

  auto torch_compute_type_a = to_torch_scalar_type<ComputeTypeA>();
  auto torch_compute_type_b = to_torch_scalar_type<ComputeTypeB>();
  auto torch_compute_type_c = to_torch_scalar_type<ComputeTypeC>();

  torch::Tensor c, out;
  bool is_gemm;

  if (!_c.has_value()) {
    auto options = torch::TensorOptions().dtype(torch_compute_type_c).device(torch::kCUDA);
    c = torch::empty({M, N}, options);
    is_gemm = true;
  } else {
    c = _c.value();
    is_gemm = false;
  }

  CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type_a)
  CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type_b)
  CHECK_TORCH_TENSOR_DTYPE(c, torch_compute_type_c)

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, N, K)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  constexpr bool IsCvtPrecision = needs_precision_conversion<ComputeTypeC, OutType>();

  if constexpr (IsCvtPrecision) {
    auto torch_compute_type_out = to_torch_scalar_type<OutType>();
    auto options = torch::TensorOptions().dtype(torch_compute_type_out).device(torch::kCUDA);
    out = torch::empty({M, N}, options);

    CHECK_TORCH_TENSOR_DTYPE(out, torch_compute_type_out)
    CHECK_TORCH_TENSOR_SHAPE(out, M, N)
  }

  using Spec = spec::KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, kBlockM, kBlockN, kBlockK, G2S_Stages>;

  dim3 block = Spec::kThreadNum;
  //dim3 grid((N + Spec::kBlockN - 1) / Spec::kBlockN, (M + Spec::kBlockM - 1) / Spec::kBlockM);
  dim3 grid(cute::ceil_div(N, Spec::kBlockN), cute::ceil_div(M, Spec::kBlockM));

  constexpr int kShmSizeA = Spec::kShmSizeA;
  constexpr int kShmSizeB = Spec::kShmSizeB;
  constexpr int kShmSizeC = Spec::kShmSizeC;
  constexpr int kShmSizeO = Spec::kShmSizeO;

  //int shm_size = Spec::kShmSize;
  int shm_size = (is_gemm) ? ((IsCvtPrecision) ? (cute::max(kShmSizeA + kShmSizeB, kShmSizeO)) : (cute::max(kShmSizeA + kShmSizeB, kShmSizeC))) 
                           : ((IsCvtPrecision) ? (cute::max(kShmSizeA + kShmSizeB + kShmSizeC, kShmSizeO)) : (kShmSizeA + kShmSizeB + kShmSizeC)); 

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
         block.z, grid.x, grid.y, grid.z, shm_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  auto get_data_ptr = [](const torch::Tensor &tensor) -> void * {
    return tensor.defined() ? tensor.data_ptr() : nullptr;
  };
  void *out_ptr = get_data_ptr(out);

  // Kernel launch
  BOOL_SWITCH(is_gemm, IsGemm, [&] {
    cudaEventRecord(start, stream);
    if (shm_size >= 48 * 1024) {
      cudaFuncSetAttribute(hgemm_cute<Spec, IsGemm, IsCvtPrecision>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shm_size);
    }
    hgemm_cute<Spec, IsGemm, IsCvtPrecision>
        <<<grid, block, shm_size, stream>>>(c.data_ptr(), a.data_ptr(), b.data_ptr(), M, N, K, out_ptr);
    cudaEventRecord(stop, stream);
  });

  cudaDeviceSynchronize();

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) +
                             " (error code: " + std::to_string(error) + ")");
  }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if constexpr (IsCvtPrecision)
    return out;
  else
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hgemm_cute_fp16_fp16_fp16_fp16",
        &(run_hgemm<128, 128, 64, 5, cute::half_t, cute::half_t, cute::half_t>),
        "Run a mixed-precision half 16x8x8 MMA operation using cute.");
  m.def("hgemm_cute_fp16_fp16_fp16_fp32",
        &(run_hgemm<128, 128, 64, 5, cute::half_t, cute::half_t, cute::half_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation using cute.");
  m.def("hgemm_cute_bf16_bf16_bf16_fp32",
        &(run_hgemm<128, 128, 64, 5, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precisioin half 16x8x8 MMA operation using cute.");
}
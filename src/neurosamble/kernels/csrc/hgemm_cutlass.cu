#include "cute/numeric/numeric_types.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <type_traits>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>

#define CUTLASS_CHECK(status)                                                                                         \
  {                                                                                                                   \
    cutlass::Status error = status;                                                                                   \
    if (error != cutlass::Status::kSuccess) {                                                                         \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl;        \
      exit(EXIT_FAILURE);                                                                                             \
    }                                                                                                                 \
  }

namespace spec {

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          typename AccType_,
          int kBlockM_,
          int kBlockN_,
          int kBlockK_,
          int G2S_Stages_ = 3>
struct KernelSpec {
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kBlockK = kBlockK_;
  static constexpr int G2S_Stages = G2S_Stages_;
  static_assert(G2S_Stages >= 2, "G2S_Stages should not be less than 2.");

  using MMA_op = std::conditional_t<
      std::is_same_v<ComputeTypeA, cute::bfloat16_t> && std::is_same_v<ComputeTypeB, cute::bfloat16_t> &&
          std::is_same_v<ComputeTypeC, float>,
      SM80_16x8x16_F32BF16BF16F32_TN,
      std::conditional_t<
          std::is_same_v<ComputeTypeA, cute::half_t> && std::is_same_v<ComputeTypeB, cute::half_t> &&
              std::is_same_v<ComputeTypeC, cute::half_t>,
          SM80_16x8x16_F16F16F16F16_TN,
          std::conditional_t<std::is_same_v<ComputeTypeA, cute::half_t> &&
                                 std::is_same_v<ComputeTypeB, cute::half_t> && std::is_same_v<ComputeTypeC, float>,
                             SM80_16x8x16_F32F16F16F32_TN,
                             void>>>;
  static_assert(!std::is_same_v<MMA_op, void>, "Unsupported MMA op!");

  using MMA_traits = MMA_Traits<MMA_op>;
  using MMA_atom = MMA_Atom<MMA_traits>;
  using MMA_shape = typename MMA_traits::Shape_MNK;

  static constexpr int kMmaThrExpandM = 2;
  static constexpr int kMmaThrExpandN = 4;
  static constexpr int kMmaThrExpandK = 1;
  static constexpr int kMmaValExpandM = 1;
  static constexpr int kMmaValExpandN = 2;
  static constexpr int kMmaValExpandK = 2;

  static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
  static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
  static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});

  using MMAThrLayout =
      decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
  using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
  using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kBlockK_Copy = cute::min(64, kBlockK) / 8;

  using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using CopyA_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeA>;
  using CopyB_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeB>;

  using TiledCopyA_G2S =
      decltype(make_tiled_copy(CopyA_G2S_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}),
                                           make_stride(Int<kBlockK_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using TiledCopyB_G2S =
      decltype(make_tiled_copy(CopyB_G2S_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}),
                                           make_stride(Int<kBlockK_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using Copy_S2R_op_A = std::conditional_t<sizeof(ComputeTypeA) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
  using Copy_S2R_op_B = std::conditional_t<sizeof(ComputeTypeB) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;

  using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op_A, ComputeTypeA>;
  using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op_B, ComputeTypeB>;

  using SmemLayoutAtomA = decltype(composition(Swizzle<3, 3, 3>{},
                                               make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockK)>{}),
                                                           make_stride(Int<cute::min(64, kBlockK)>{}, Int<1>{}))));
  using SmemLayoutAtomB = SmemLayoutAtomA;

  // A matrix configuration
  using ElementA = ComputeTypeA_;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 16 / sizeof(ElementA);

  // B matrix configuration.
  // The SM80 *_TN MMA atom contracts over K and needs BOTH operands K-major.
  // For CUTLASS's B operand (logical shape [N, K]) "K-major" == ColumnMajor.
  // RowMajor would make B N-major and break the 128-bit cp.async along K.
  using ElementB = ComputeTypeB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 16 / sizeof(ElementB);

  using ElementC = ComputeTypeC_;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 16 / sizeof(ElementC);

  using ElementD = OutType_;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using ElementAccumulator = AccType_;
  using ElementCompute = AccType_;
  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<Int<kBlockM_>, Int<kBlockN_>, Int<kBlockK_>>;

  using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<G2S_Stages>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<DispatchPolicy,
                                                                      TileShape,
                                                                      ElementA,
                                                                      cutlass::gemm::TagToStrideA_t<LayoutA>,
                                                                      ElementB,
                                                                      cutlass::gemm::TagToStrideB_t<LayoutB>,
                                                                      TiledMMA,
                                                                      TiledCopyA_G2S,
                                                                      SmemLayoutAtomA,
                                                                      CopyA_S2R_atom,
                                                                      cute::identity,
                                                                      TiledCopyB_G2S,
                                                                      SmemLayoutAtomB,
                                                                      CopyB_S2R_atom,
                                                                      cute::identity>;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      ElementC,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      cutlass::epilogue::thread::LinearCombination<ElementD,
                                                   1,
                                                   ElementAccumulator,
                                                   ElementCompute,
                                                   cutlass::epilogue::thread::ScaleType::Default,
                                                   cutlass::FloatRoundStyle::round_to_nearest,
                                                   ElementC>,
      cutlass::gemm::EpilogueDefault>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  static void run(void *Aptr, void *Bptr, void *Cptr, void *Dptr, int M, int N, int K, cudaStream_t stream = nullptr) {
    Gemm gemm;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo kernel_hw_info;
    kernel_hw_info.device_id = 0;
    kernel_hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(kernel_hw_info.device_id);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {(ElementA *)Aptr, stride_A, (ElementB *)Bptr, stride_B},
        // alpha = 1, beta = 0 : plain GEMM output, do NOT add the (uninitialized) C source.
        {{(ElementAccumulator)1.f, (ElementAccumulator)0.f}, (ElementC *)Cptr, stride_C, (ElementD *)Dptr, stride_D},
        kernel_hw_info};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run(stream));
  }
};

} // namespace spec

// Explicit instantiation of the FP16 kernel.
template struct spec::KernelSpec<cutlass::half_t, cute::half_t, cute::half_t, float, float, 128, 128, 32>;

// Python-facing entry point. Contract (unchanged): C = A @ B, all row-major.
//   A : [M, K] row-major
//   B : [K, N] row-major
//   C : [M, N] row-major
// The *_TN tensor-core kernel needs B K-major, but a row-major [K, N] B is
// N-major, so we transpose it to [N, K] (K-major) before launching. This keeps
// the gemm(A, B) = A @ B contract, so ops.py needs no changes.
torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    A = A.contiguous();
    B = B.contiguous();
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);   // B is [K, N]

    // FP32 path: use PyTorch's native GEMM.
    if (A.scalar_type() == torch::kFloat32) {
        return at::mm(A, B);
    }

    // FP16 path: CUTLASS.
    if (A.scalar_type() == torch::kFloat16) {
        // [K, N] (N-major) -> [N, K] (K-major) for the TN kernel.
        auto Bt = B.transpose(0, 1).contiguous();
        auto options = torch::TensorOptions().dtype(A.scalar_type()).device(A.device());
        torch::Tensor C = torch::empty({M, N}, options);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        using Kernel = spec::KernelSpec<cutlass::half_t, cute::half_t, cute::half_t, float, float, 128, 128, 32>;
        Kernel::run(A.data_ptr(), Bt.data_ptr(), C.data_ptr(), C.data_ptr(), M, N, K, stream);
        cudaStreamSynchronize(stream);
        return C;
    }

    throw std::runtime_error("Unsupported dtype for CUTLASS gemm");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "CUTLASS GEMM: C = A @ B");
}

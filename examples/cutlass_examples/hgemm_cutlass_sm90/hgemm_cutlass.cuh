#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/types.h>

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

template<typename OutType_,
         typename ComputeTypeA_,
         typename ComputeTypeB_,
         typename ComputeTypeC_,
         typename AccType_,
         int kBlockM_,
         int kBlockN_,
         int kBlockK_>
    struct KernelSpec{

        // A matrix configuration
        using ElementA = ComputeTypeA_;
        using LayoutA = cutlass::layout::RowMajor;
        static constexpr int AlignmentA = 16 / sizeof(ElementA);
        // static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

        // B matrix configuration
        using ElementB = ComputeTypeB_;
        using LayoutB = cutlass::layout::ColumnMajor;
        static constexpr int AlignmentB = 16 / sizeof(ElementB);

        // C/D matrix configuration
        using ElementC = ComputeTypeC_;
        using LayoutC = cutlass::layout::RowMajor;
        static constexpr int AlignmentC = 16 / sizeof(ElementC);

        using ElementD = OutType_;
        using LayoutD = cutlass::layout::RowMajor;
        static constexpr int AlignmentD = 16 / sizeof(ElementD);

        // Core kernel configurations
        using ElementAccumulator = AccType_;
        using ElementCompute = AccType_;
        using ArchTag = cutlass::arch::Sm120;
        using OperatorClass = cutlass::arch::OpClassTensorOp;
        using TileShape = Shape<Int<kBlockM_>, Int<kBlockN_>, Int<kBlockK_>>;
        using ClusterShape = Shape<_2, _1, _1>;
        using StageCount = cutlass::gemm::collective::StageCountAuto; 
        using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
        using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
        using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
        using FusionOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute>;
        using TileScheduler = cutlass::gemm::PersistentScheduler;

        // template <
        //   class ArchTag, class OpClass,
        //   class TileShape_MNK, class ClusterShape_MNK,
        //   class EpilogueTileType,
        //   class ElementAccumulator, class ElementCompute,
        //   class ElementC, class GmemLayoutTagC, int AlignmentC,
        //   class ElementD, class GmemLayoutTagD, int AlignmentD,
        //   class EpilogueScheduleType,
        //   class FusionOpOrCallbacks =
        //   cutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>, class Enable =
        //   void
        // >
        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, 
                                                                                             OperatorClass,
                                                                                             TileShape,
                                                                                             ClusterShape,
                                                                                             EpilogueTile,
                                                                                             ElementAccumulator,
                                                                                             ElementCompute,
                                                                                             ElementC,
                                                                                             LayoutC,
                                                                                             AlignmentC,
                                                                                             ElementD,
                                                                                             LayoutD,
                                                                                             AlignmentD,
                                                                                             EpilogueSchedule,
                                                                                             FusionOperation>;
        // template <
        //   class ArchTag, class OpClass,
        //   class ElementA, class GmemLayoutA, int AlignmentA,
        //   class ElementB, class GmemLayoutB, int AlignmentB,
        //   class ElementAccumulator,
        //   class TileShape_MNK, class ClusterShape_MNK,
        //   class StageCountType,
        //   class KernelScheduleType,
        //   class Enable = void
        // >
        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, 
                                                                                         OperatorClass, 
                                                                                         ElementA, 
                                                                                         LayoutA, 
                                                                                         AlignmentA, 
                                                                                         ElementB, 
                                                                                         LayoutB, 
                                                                                         AlignmentB, 
                                                                                         ElementAccumulator, 
                                                                                         TileShape, 
                                                                                         ClusterShape, 
                                                                                         conditional<cute::is_same_v<StageCount, cutlass::gemm::collective::StageCountAuto>,
                                                                                                    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, 
                                                                                         StageCount>, KernelSchedule>::collectiveOp;                                                                                    

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int>, 
                                                                CollectiveMainloop, 
                                                                CollectiveEpilogue, 
                                                                TileScheduler>;

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

        using StrideA = typename Gemm::GemmKernel::StrideA;
        using StrideB = typename Gemm::GemmKernel::StrideB;
        using StrideC = typename Gemm::GemmKernel::StrideC;
        using StrideD = typename Gemm::GemmKernel::StrideD;

        static void run(void *Aptr, void *Bptr, void *Cptr, void *Dptr, int M, int N, int K, cudaStream_t stream = nullptr) {
            // Instantiate CUTLASS kernel depending on templates
            Gemm gemm;

            // Make strides
            StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
            StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
            StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
            StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

            // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm

            // Change device_id to another value if you are running on a machine with multiple GPUS and  wish to use a GPU other than that with deive ID 0;
            int device_id = 0;
            cutlass::KernelHardwareInfo kernel_hw_info = 
                cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);

            typename Gemm::Arguments arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {M, N, K},
                {(ElementA *)Aptr, stride_A, (ElementB *)Bptr, stride_B},
                {{(ElementAccumulator)1.f, (ElementAccumulator)1.f}, (ElementC *)Cptr, stride_C, (ElementD *)Dptr, stride_D},
                kernel_hw_info
            };

            // Using the arguments, query for extra workspace required for matrix multiplication computation
            size_t workspace_size = Gemm::get_workspace_size(arguments);

            // Allocate worksapece memory
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

            // Check if the problem size is supported or not
            CUTLASS_CHECK(gemm.can_implement(arguments));

            // Initialize CUTLASS kernel with arguments and workspace pointer
            CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

            // Correctness / Warmup interation
            CUTLASS_CHECK(gemm.run(stream));
        }
    };
    
}

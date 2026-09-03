#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && ((__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 4)))
#define CUTE_ARCH_MMA_SM89_ENABLED
#endif

#if defined(__CUDA_ARCH__)
#define CUTE_INVALID_CONTROL_PATH(x)                                                                                  \
  assert(0 && x);                                                                                                     \
  printf(x);                                                                                                          \
  __brkpt()
#else
#define CUTE_INVALID_CONTROL_PATH(x)                                                                                  \
  assert(0 && x);                                                                                                     \
  printf(x)
#endif

namespace cute {

struct SM90_16x8x32_F32E4M3E5M2F32_TN{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float &d0,
                                     float &d1,
                                     float &d2, 
                                     float &d3,
                                     uint32_t const &a0,
                                     uint32_t const &a1,
                                     uint32_t const &a2,
                                     uint32_t const &a3,
                                     uint32_t const &b0,
                                     uint32_t const &b1,
                                     float const &c0,
                                     float const &c1,
                                     float const &c2,
                                     float const &c3) {
#if defined(CUTE_ARCH_MMA_SM89_ENABLED)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else 
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM90_16x8x32_F32E4M3E5M2F32_TN without CUTE_ARCH_MMA_SM89_ENABLED");
#endif

    }
};

template<> struct MMA_Traits<SM90_16x8x32_F32E4M3E5M2F32_TN> {
    using ValTypeD = float;
    using ValTypeA = float_e4m3_t;
    using ValTypeB = float_e5m2_t;
    using ValTypeC = float;

    using Shape_MNK = Shape<_16, _8, _32>;
    using ThrID = Layout<_32>;
    using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2, _2>>, Stride<Stride<_64, _1>, Stride<_16, _8, _256>>>; 
    using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>, Stride<Stride<_32, _1>, Stride<_8, _128>>>;
    using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>, Stride<Stride<_32, _1>, Stride<_16, _8>>>;

};

};  //namespace cute

namespace spec {

using namespace cute;
template<typename OutType_, 
         typename ComputeTypeA_,
         typename ComputeTypeB_,
         typename ComputeTypeC_,
         int kTileM_ = 16,
         int kTileN_ = 8,
         int kTileK_ = 8>
struct KernelSpec{
    using OutType = OutType_;
    using ComputeTypeA = ComputeTypeA_;
    using ComputeTypeB = ComputeTypeB_;
    using ComputeTypeC = ComputeTypeC_;

    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;

    using MMA_op = std::conditional_t<std::is_same_v<ComputeTypeA, bfloat16_t> && std::is_same_v<ComputeTypeB, bfloat16_t> && std::is_same_v<ComputeTypeC, float>, 
                                SM80_16x8x8_F32BF16BF16F32_TN,
                                     std::conditional_t<std::is_same_v<ComputeTypeA, float_e4m3_t> && std::is_same_v<ComputeTypeB, float_e5m2_t> && std::is_same_v<ComputeTypeC, float>, 
                                SM90_16x8x32_F32E4M3E5M2F32_TN,
                                void>>;

    static_assert(!std::is_same_v<MMA_op, void>, "Unsupported MMA op!");

    using TiledMMA = decltype(make_tiled_mma(MMA_op{}));

    static constexpr int kThreadNum = size(TiledMMA{});
    static constexpr int kShmSize = 0;
};

};  // namespace spec

template<typename Spec, bool IsGemm, bool IsCvPrecision>
__global__ void 
mixed_precision_gemm(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k, void *OutPtr) {
    using namespace cute;

    using X = Underscore;
    using OutType = typename Spec::OutType;
    using ComputeTypeA = typename Spec::ComputeTypeA;
    using ComputeTypeB = typename Spec::ComputeTypeB;
    using ComputeTypeC = typename Spec::ComputeTypeC;
    using TiledMMA = typename Spec::TiledMMA;

    constexpr int kTileM = Spec::kTileM;
    constexpr int kTileN = Spec::kTileN;
    constexpr int kTileK = Spec::kTileK;

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA*)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB*)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC*)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
    Tensor m0 = make_tensor(make_gmem_ptr((OutType*)OutPtr), make_shape(m, n), make_stride(n, Int<1>{}));

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});
    Tensor g0 = local_tile(m0, tiler, coord, Step<_1, _1, X>{});

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.partition_fragment_A(gA);
    Tensor tCrB = thr_mma.partition_fragment_B(gB);
    Tensor tCrC = thr_mma.partition_fragment_C(gC);

    auto copy_atom = AutoVectorizingCopy{};

    copy(copy_atom, tCgA, tCrA);
    copy(copy_atom, tCgB, tCrB);

    if constexpr (IsGemm) {
        clear(tCrC);
    }else {
        copy(copy_atom, tCgC, tCrC);
    }

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    if constexpr (!IsCvPrecision) {
        copy(copy_atom, tCrC, tCgC);
    }else {
        auto tCr0 = make_tensor_like<OutType>(tCrC); 
        copy(tCrC, tCr0);

        Tensor tCg0 = thr_mma.partition_C(g0);
        copy(copy_atom, tCr0, tCg0);
    }
}
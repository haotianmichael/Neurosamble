#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#if (defined (__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_ENABLED
#endif

namespace spec{
    using namespace cute;

    template<typename OutType_,
             typename ComputeTypeA_,
             typename ComputeTypeB_,
             typename ComputeTypeC_,
             int kBlockM_,
             int kBlockN_,
             int kBlockK_,
             int G2S_Stages_ = 3>
    struct KernelSpec{
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
        static constexpr int kMmaValExpandN = 1;
        static constexpr int kMmaValExpandK = 2;

        static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
        static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
        static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});
        
        // (M, N, K)->warp_idx(MMA Atom)
        using MMAThrLayout = decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
        // Permutation Layout
        using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
        using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

      // Why AutoVectorizingCopy faults under copy_if (CUDA error 716, "misaligned
      // address"):
      //
      //   `AutoVectorizingCopyWithAssumedAlignment<MaxBits>` inherits from
      //   `UniversalCopy<uint_bit_t<MaxBits>>`, so `AutoVectorizingCopy` (=
      //   `<128>`) is structurally a 128-bit atom applied to whatever element
      //   type the tensor has.
      //
      //   In `cute/algorithm/copy.hpp`, the `copy()` overload for AutoVec does a
      //   `recast<uint_bit_t<vec_bits>>` of src/dst BEFORE issuing the atom:
      //
      //       copy(AutoVec<N>, src, dst)
      //         -> recast<uintN>(src/dst)        // fp16 -> uint128_t view
      //         -> copy_if(true, src_v, dst_v)   // 1 atom call per iter
      //
      //   But `copy_if(Copy_Atom<AutoVec<N>, T>, pred, src, dst)` has NO matching
      //   recast specialization. It falls through to the generic Copy_Atom
      //   path that unrolls atom calls over the per-thread tile WITHOUT
      //   recasting. With val (1, 8) of fp16 we get 8 atom calls at stride
      //   1 fp16 (= 2 B), each invoking the 128-bit atom:
      //
      //       ld.global.nc.v2.u64 [base + 0]    aligned
      //       ld.global.nc.v2.u64 [base + 2]    misaligned -> fault
      //       ld.global.nc.v2.u64 [base + 4]    misaligned
      //       ...
      //       ld.global.nc.v2.u64 [base + 14]   misaligned
      //
      //   These loads also have NO @p in PTX: copy_if's `if (pred)` becomes one
      //   coarse `setp + @p bra` gating the whole block. NVCC faithfully lowers
      //   exactly what CUTLASS asked for; the broken stride is at the CUTLASS
      //   layer, not at NVCC.
      //
      // Tracked upstream: NVIDIA/cutlass#2354 ("Missing copy_if implementation
      // for AutoVectorizingCopyWithAssumedAlignment"). As of CUTLASS main the
      // bug is still present -- no copy_if(AutoVec<N>, ...) specialization has
      // been merged. The official workaround in
      // cutlass/examples/cute/tutorial/tiled_copy_if.cu is to bake the width
      // into the atom type explicitly:
      //
      //   using CopyOp = UniversalCopy<uint_byte_t<sizeof(T) * size(val_layout)>>;
      //
      // SM80_CP_ASYNC_CACHEGLOBAL<uint128_t> follows the same principle: its
      // atom is 16 B intrinsically, so val (1, 8) collapses to one atom call per
      // iter -> one aligned `cp.async.cg.shared.global [smem], [gmem], 16` per
      // thread. It also drives the cp_async_fence / cp_async_wait pipeline
      // (AutoVec is sync ld+st, the fences become no-ops with respect to data
      // motion).
        using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

        // Note: ldmatrix only support 16-bit data type(or below)
        using Copy_S2R_op_A = std::conditional_t<sizeof(ComputeTypeA) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
        using Copy_S2R_op_B = std::conditional_t<sizeof(ComputeTypeB) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
        // The C / O accumulator fragment has no K val-expand, so each thread only
        // owns half the 32-bit packets of an A/B fragment. Use the x2 LDSM/STSM
        // variant for C-side copies; A/B keep x4 because VAL_EXPAND_K=2 gives them
        // enough vals/thread.
        using Copy_S2R_op_C = std::conditional_t<sizeof(ComputeTypeC) == 2, SM75_U32x2_LDSM_N, AutoVectorizingCopy>;

        using CopyA_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeA>;
        using CopyB_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeB>;
        using CopyC_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeC>;

        using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op_A, ComputeTypeA>;
        using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op_B, ComputeTypeB>;
        using CopyC_S2R_atom = Copy_Atom<Copy_S2R_op_C, ComputeTypeC>;

    #if (defined (__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        // R2S writes a C-shaped fragment, so match the C-side x2 variant chosen
        // above. SM90_U32x4_STSM_N would static_assert on too-few vals/thread.
        using Copy_R2S_op = SM90_U32x2_STSM_N;
    #else
        using Copy_R2S_op = AutoVectorizingCopy;
    #endif

        using Copy_S2G_op = UniversalCopy<cute::uint128_t>;

        using CopyC_R2S_atom = Copy_Atom<Copy_R2S_op, ComputeTypeC>;
        using CopyO_R2S_atom = Copy_Atom<Copy_R2S_op, OutType>;

        using CopyC_S2G_atom = Copy_Atom<Copy_S2G_op, ComputeTypeC>;
        using CopyO_S2G_atom = Copy_Atom<Copy_S2G_op, OutType>;


        static constexpr int kThreadNum = size(TiledMMA{});
        static constexpr int kBlockK_Copy = cute::min(64, kBlockK) / 8;
        static constexpr int kBlockN_Copy = cute::min(64, kBlockN) / 8;

        using TiledCopyA_G2S = decltype(make_tiled_copy(CopyA_G2S_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}), make_stride(Int<kBlockK_Copy>{}, Int<1>{})), 
                make_layout(make_shape(Int<1>{}, Int<8>{}))));

        using TiledCopyB_G2S = decltype(make_tiled_copy(CopyB_G2S_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}), make_stride(Int<kBlockK_Copy>{}, Int<1>{})), 
                make_layout(make_shape(Int<1>{}, Int<8>{}))));

        using TiledCopyC_G2S = decltype(make_tiled_copy(CopyC_G2S_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}), make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                make_layout(make_shape(Int<1>{}, Int<8>{}))));

        using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_S2R_atom{}, TiledMMA{}));
        using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_S2R_atom{}, TiledMMA{}));
        using TiledCopyC_S2R = decltype(make_tiled_copy_C(CopyC_S2R_atom{}, TiledMMA{}));

        using TiledCopyC_R2S = decltype(make_tiled_copy_C(CopyC_R2S_atom{}, TiledMMA{}));
        using TiledCopyO_R2S = decltype(make_tiled_copy_C(CopyO_R2S_atom{}, TiledMMA{}));

        using TiledCopyC_S2G = decltype(make_tiled_copy(CopyC_S2G_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}), make_stride(Int<kBlockN_Copy>{}, Int<1>{})), 
                make_layout(make_shape(Int<1>{}, Int<8>{}))));
        using TiledCopyO_S2G = decltype(make_tiled_copy(CopyO_S2G_atom{}, 
                    make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}), make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                make_layout(make_shape(Int<1>{}, Int<8>{})))); 


        using SmemLayoutAtomA = decltype(composition(Swizzle<3, 3, 3>{},
                                                make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockK)>{}),
                                                            make_stride(Int<cute::min(64, kBlockK)>{}, Int<1>{}))));
        using SmemLayoutAtomB = decltype(composition(Swizzle<3, 3, 3>{},
                                               make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockK)>{}),
                                                           make_stride(Int<cute::min(64, kBlockK)>{}, Int<1>{}))));
        using SmemLayoutAtomC = decltype(composition(Swizzle<3, 3, 3>{},
                                               make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockN)>{}),
                                                           make_stride(Int<cute::min(64, kBlockN)>{}, Int<1>{}))));
        using SmemLayoutAtomO = decltype(composition(Swizzle<3, 3, 3>{},
                                               make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockN)>{}),
                                                           make_stride(Int<cute::min(64, kBlockN)>{}, Int<1>{}))));
                                                           /*using SmemLayoutA = decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}), make_stride(Int<kBlockK>{}, Int<1>{})));
        using SmemLayoutB = decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kBlockK>{}), make_stride(Int<kBlockK>{}, Int<1>{})));
        using SmemLayoutC = decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, Int<1>{})));
        using SmemLayoutO = decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, Int<1>{})));*/

        using SmemLayoutA =
            decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<G2S_Stages>{})));
         using SmemLayoutB =
            decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<kBlockN>{}, Int<kBlockK>{}, Int<G2S_Stages>{})));
         using SmemLayoutC =
            decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
         using SmemLayoutO =
            decltype(tile_to_shape(SmemLayoutAtomO{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));

        static constexpr int kShmSizeA = cosize(SmemLayoutA{}) * sizeof(ComputeTypeA);
        static constexpr int kShmSizeB = cosize(SmemLayoutB{}) * sizeof(ComputeTypeB);
        static constexpr int kShmSizeC = cosize(SmemLayoutC{}) * sizeof(ComputeTypeC);
        static constexpr int kShmSizeO = cosize(SmemLayoutO{}) * sizeof(OutType);

        //static constexpr int kShmSize = cute::max( kShmSizeA + kShmSizeB + kShmSizeC, kShmSizeO);
    };
}

template<typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ __launch_bounds__(Spec::kThreadNum) void hgemm_cute_no_reg_prefetch(void *__restrict__ Cptr, const void *__restrict__ Aptr, const void *__restrict__ Bptr, int m, int n, int k, void *__restrict__ Outptr) {

    using namespace cute;

    using X = Underscore;
    using MMA_shape = typename Spec::MMA_shape;
    using OutType = typename Spec::OutType;
    using ComputeTypeA = typename Spec::ComputeTypeA;
    using ComputeTypeB = typename Spec::ComputeTypeB;
    using ComputeTypeC = typename Spec::ComputeTypeC;
    using SmemLayoutA = typename Spec::SmemLayoutA;
    using SmemLayoutB = typename Spec::SmemLayoutB;
    using SmemLayoutC = typename Spec::SmemLayoutC;
    using SmemLayoutO = typename Spec::SmemLayoutO;

    constexpr int kBlockM = Spec::kBlockM;
    constexpr int kBlockN = Spec::kBlockN;
    constexpr int kBlockK = Spec::kBlockK;
    constexpr int kShmSizeA = Spec::kShmSizeA;
    constexpr int kShmSizeB = Spec::kShmSizeB;
    constexpr int G2S_Stages = Spec::G2S_Stages;

    extern __shared__ __align__(1024) uint8_t smem[];

    uint8_t *Aptr_smem = smem;
    uint8_t *Bptr_smem = smem + kShmSizeA;
    uint8_t *Cptr_smem;
    if constexpr (!IsGemm)
        Cptr_smem = smem + kShmSizeA + kShmSizeB;
    else 
        Cptr_smem = smem;
    uint8_t *Optr_smem = smem;

    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA*)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));   // (M, K)       
    Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB*)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));  // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC*)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));  // (M, N)
    Tensor m0 = make_tensor(make_gmem_ptr((OutType*)Outptr), make_shape(m, n), make_stride(n, Int<1>{}));  // (M, N)

    auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
    auto coord = make_coord(bidy, bidx, _);  

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});   // (BLK_M, BLK_K, K_TILES)
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});   // (BLK_N, BLK_K, K_TILES)
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});   // (BLK_M, BLK_N)
    Tensor g0 = local_tile(m0, tiler, coord, Step<_1, _1, X>{});  // (BLK_M, BLK_N)

    auto m_max_coord = m - size<0>(gA) * bidy; // M - BLK_M * m_coord
    auto n_max_coord = n - size<0>(gB) * bidx;  // N - BLK_N * n_coord
    auto k_residue = k - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    gA = domain_offset(make_coord(0, k_residue, 0), gA);
    gB = domain_offset(make_coord(0, k_residue, 0), gB);


    Tensor sA = make_tensor(make_smem_ptr((ComputeTypeA*)Aptr_smem), SmemLayoutA{});  // (BLK_M, BLK_K, G2S_PIPE)
    Tensor sB = make_tensor(make_smem_ptr((ComputeTypeB*)Bptr_smem), SmemLayoutB{});  // (BLK_N, BLK_K, G2S_PIPE)
    Tensor sC = make_tensor(make_smem_ptr((ComputeTypeC*)Cptr_smem), SmemLayoutC{}); // (BLK_M, BLKN)
    Tensor s0 = make_tensor(make_smem_ptr((OutType*)Optr_smem), SmemLayoutO{}); // (BLK_M, BLK_N)

    typename Spec::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    //Tensor tCgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K)
    //Tensor tCgB = thr_mma.partition_B(gB);
    //Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K) 
    Tensor tCrC = thr_mma.partition_fragment_C(gC); // (MMA, MMA_M, MMA_N)

    //--- Copy all global matrix Tile A/B/C to SMEM
    typename Spec::TiledCopyA_G2S g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA); // (ACPY, ACPY_M, ACPY_K, K_TILES)
    Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA); // (ACPY, ACPY_M, ACPY_K, G2S_PIPE)

    typename Spec::TiledCopyB_G2S g2s_tiled_copy_b;
    ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB); // (BCPY, BCPY_N, BCPY_K, K_TILES)
    Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB); // (BCPY, BCPY_N, BCPY_K, G2S_PIPE)

    typename Spec::TiledCopyC_G2S g2s_tiled_copy_c;
    ThrCopy g2s_thr_copy_c = g2s_tiled_copy_c.get_slice(tid);
    Tensor tCgC_g2s = g2s_thr_copy_c.partition_S(gC); // (CCPY, CCPY_M, CCPY_N)
    Tensor tCsC_g2s = g2s_thr_copy_c.partition_D(sC); // (CCPY, CCPY_M, CCPY_N)

    //
    // PREDICATES
    //

    // Allocate predicate tensors
    Tensor tApA_g2s = make_tensor<bool>(make_shape(size<1>(tAsA_g2s), size<2>(tAsA_g2s)), Stride<_1, _0>{});  // (ACPY_M, ACPY_K)
    Tensor tBpB_g2s = make_tensor<bool>(make_shape(size<1>(tBsB_g2s), size<2>(tBsB_g2s)), Stride<_1, _0>{}); // (BCPY_N, BCPY_K)
    Tensor tCpC_g2s = make_tensor<bool>(make_shape(size<1>(tCsC_g2s), size<2>(tCsC_g2s)), Stride<_1, Int<size<1>(tCsC_g2s)>>{});  // (CCPY_M, CCPY_N)

    // Construct identity layout
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));  // (BLK_M, BLK_K) -> (blk_m, blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));  // (BLK_N, BLK_K) -> (blk_n, blk_k)
    Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC)));  // (BLK_M, BLK_N) -> (blk_m, blk_n)

    // Repeat the partitioning with identity layouts
    Tensor tAcA_g2s = g2s_thr_copy_a.partition_S(cA);  // (ACPY, ACPY_M, ACPY_K) -> (blk_m, blk_k)
    Tensor tBcB_g2s = g2s_thr_copy_b.partition_S(cB);  // (BCPY, BCPY_N, BCPY_K) -> (blk_n, blk_k)
    Tensor tCcC_g2s = g2s_thr_copy_c.partition_S(cC);  // (CCPY, CCPY_M, CCPY_N) -> (blk_m, blk_n)

    // Set predicates for m bounds
#pragma unroll
    for(int m = 0; m < size<0>(tApA_g2s); ++m) {
        tApA_g2s(m, 0) = get<0>(tAcA_g2s(0, m, 0)) < m_max_coord;  // blk_m coord < residue_m
    }
    // Set predicates for n bounds
#pragma unroll
    for(int n = 0; n < size<0>(tBpB_g2s); ++n) {
        tBpB_g2s(n, 0) = get<0>(tBcB_g2s(0, n, 0)) < n_max_coord;  // blk_n coord < residue_n
    }
    // Set predicates for (m, n) bounds
#pragma unroll 
    for(int m = 0; m < size<0>(tCpC_g2s); ++m) {
#pragma unroll 
        for(int n = 0; n < size<1>(tCpC_g2s); ++n) {
            // blk_m coord < residue_m and blk_n < residue_n
            tCpC_g2s(m, n) = elem_less(tCcC_g2s(0, m, n), make_coord(m_max_coord, n_max_coord));
            // Equivalent to:
            // tCpC_g2s(m,n) = (get<0>(tCcC_g2s(0,m,n)) < m_max_coord) && (get<1>(tCcC_g2s(0,m,n)) < n_max_coord);
        }
    }

    //
    // END PREDICATES
    //


    /*Naive Copy 
    copy(g2s_tiled_copy_a, tAgA_g2s, tAsA_g2s);
    copy(g2s_tiled_copy_b, tBgB_g2s, tBsB_g2s);
    if constexpr (!IsGemm) {
        copy(g2s_tiled_copy_c, tCgC_g2s, tCsC_g2s);
    }

    #if defined(CP_ASYNC_ENABLED)
        cp_async_fence();
        cp_async_wait<0>();
    #endif
    __syncthreads();*/
    //--- Complete copy from GMEM to SMEM

    typename Spec::TiledCopyA_S2R s2r_tiled_copy_a;
    ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA);
    Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);

    typename Spec::TiledCopyB_S2R s2r_tiled_copy_b;
    ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
    Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB);
    Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);


    typename Spec::TiledCopyC_S2R s2r_tiled_copy_c;
    ThrCopy s2r_thr_copy_c = s2r_tiled_copy_c.get_slice(tid);
    Tensor tCsC_s2r = s2r_thr_copy_c.partition_S(sC);
    Tensor tCrC_s2r = s2r_thr_copy_c.retile_D(tCrC);

    // 
    // Prefetch
    //

    if constexpr (!IsGemm) {
        // Clear the smem tiles to account for predicated off loads
        clear(tCsC_g2s);
        copy_if(g2s_tiled_copy_c, tCpC_g2s, tCgC_g2s, tCsC_g2s);
    }

    int NTilesK = ceil_div(k, kBlockK);
     // Zero the A/B smem ONCE before the K-loop (hoisted out of the mainloop).
     // The predicated cp.async never writes masked-off slots, so they must read
     // as 0. A single clear suffices: the M-boundary rows masked off by the
     // M-only predicate are never written by any K-tile, and the ik=0 K-residue
     // columns are overwritten by every later (full) K-tile.
     //
     // No sync between the clear and the cp.async: each thread's clear targets
     // the same g2s partition slots that its cp.async then writes, so program
     // order within a thread suffices. The __syncthreads() after cp_async_wait
     // below publishes both the zeros and the cp.async data to the s2r readers.
     clear(tAsA_g2s);
     clear(tBsB_g2s);

#pragma unroll
    for (int k = 0; k < size<2>(tAsA_g2s); ++k) {
      if (get<1>(tAcA_g2s(0, 0, k)) >= -k_residue) { // blk_k coord < residue_k (gA shifted)
        copy_if(g2s_tiled_copy_a, tApA_g2s(_, k), tAgA_g2s(_, _, k, 0), tAsA_g2s(_, _, k, 0));
      }
    }

#pragma unroll
    for (int k = 0; k < size<2>(tBsB_g2s); ++k) {
      if (get<1>(tBcB_g2s(0, 0, k)) >= -k_residue) { // blk_k coord < residue_k (gB shifted)
        copy_if(g2s_tiled_copy_b, tBpB_g2s(_, k), tBgB_g2s(_, _, k, 0), tBsB_g2s(_, _, k, 0));
      }
    }

    cp_async_fence();

#pragma unroll
    for (int ik = 1; ik < G2S_Stages - 1; ++ik) {
      // Set all predicates to false if we are going to overshoot bounds
      if (ik == NTilesK) {
        clear(tApA_g2s);
        clear(tBpB_g2s);
      }

      copy_if(g2s_tiled_copy_a, tApA_g2s, tAgA_g2s(_, _, _, ik), tAsA_g2s(_, _, _, ik));
      copy_if(g2s_tiled_copy_b, tBpB_g2s, tBgB_g2s(_, _, _, ik), tBsB_g2s(_, _, _, ik));

      cp_async_fence();
    }

    cp_async_wait<G2S_Stages - 2>();
    __syncthreads();

    //
    // MAINLOOP
    //

    int g2s_gmem_pipe = G2S_Stages - 1;
    int g2s_smem_pipe = G2S_Stages - 1;
    int s2r_smem_pipe = 0;

    for(int ik = 0; ik < NTilesK; ik ++) {

        copy(s2r_tiled_copy_a, tAsA_s2r(_, _, _, s2r_smem_pipe), tArA_s2r);
        copy(s2r_tiled_copy_b, tBsB_s2r(_, _, _, s2r_smem_pipe), tBrB_s2r);

        {
            // Set all predicates to false if we are going to overshoot bounds
            if(g2s_gmem_pipe == NTilesK) {
                clear(tApA_g2s);
                clear(tBpB_g2s);
            }

            copy_if(g2s_tiled_copy_a, tApA_g2s, tAgA_g2s(_, _, _, g2s_gmem_pipe), tAsA_g2s(_, _, _, g2s_smem_pipe));
            copy_if(g2s_tiled_copy_b, tBpB_g2s, tBgB_g2s(_, _, _, g2s_gmem_pipe), tBsB_g2s(_, _, _, g2s_smem_pipe));
      
            cp_async_fence();
            ++g2s_gmem_pipe;
            ++g2s_smem_pipe;
            g2s_smem_pipe = (g2s_smem_pipe == G2S_Stages) ? 0 : g2s_smem_pipe;
        }
        if (ik == 0) {
            if constexpr (IsGemm) {
                clear(tCrC); // Set the accumulators to zero
            } else {
                copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r);
            }
        }

        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

        cp_async_wait<G2S_Stages - 2>();
        __syncthreads();
        ++s2r_smem_pipe;
        s2r_smem_pipe = (s2r_smem_pipe == G2S_Stages) ? 0 : s2r_smem_pipe;
    }

    /*naive copy
    #if 1
        copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
        copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
    #else
        constexpr int kMmaValExpandM = Spec::kMmaValExpandM;
        constexpr int kMmaValExpandN = Spec::kMmaValExpandN;
        constexpr int kMmaValExpandK = Spec::kMmaValExpandK;

        constexpr int kMmaTileM = Spec::kMmaTileM;
        constexpr int kMmaTileN = Spec::kMmaTileN;
        constexpr int kMmaTileK = Spec::kMmaTileK;

        constexpr int NTilesM = kBlockM / kMmaTileM; // 4
        constexpr int NTilesN = kBlockN / kMmaTileN; // 4
        constexpr int NTilesK = kBlockK / kMmaTileK; // 2

    #pragma unroll
        for(int m_tile = 0; m_tile < NTilesM; ++m_tile) {
    #pragma unroll
            for(int n_tile = 0; n_tile < NTilesN; ++n_tile) {
    #pragma unroll
                for(int k_tile = 0; k_tile < NTilesK; ++k_tile) {
    #pragma unroll
                copy(s2r_tiled_copy_a, tAsA_s2r(_, m_tile, k_tile), tArA_s2r(_, m_tile, k_tile));
                copy(s2r_tiled_copy_b, tBsB_s2r(_, n_tile, k_tile), tBrB_s2r(_, n_tile, k_tile));
    #pragma unroll
                    for(int im = m_tile * kMmaValExpandM; im < (m_tile + 1) * kMmaValExpandM; ++im) {
    #pragma unroll
                        for(int in = n_tile * kMmaValExpandN; in < (n_tile + 1) * kMmaValExpandN; ++in) {
    #pragma unroll 
                            for(int ik = k_tile * kMmaValExpandK; ik < (k_tile + 1) * kMmaValExpandK; ++ik) {
                                gemm(tiled_mma, tCrC(_, im, in), tCrA(_, im, ik), tCrB(_, in, ik), tCrC(_, im, in));
                            }
                        }
                    }
                }
            }
        }

    #endif
    __syncthreads();*/

    cp_async_wait<0>();
    __syncthreads();

    //
    // EPILOGUE
    //

    if constexpr (!IsCvtPrecision) {
        typename Spec::TiledCopyC_R2S r2s_tiled_copy_c;
        ThrCopy r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(tid);
        Tensor tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);
        Tensor tCsC_r2s = r2s_thr_copy_c.partition_D(sC);
        copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);

        __syncthreads();

        typename Spec::TiledCopyC_S2G s2g_tiled_copy_c;
        ThrCopy s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(tid);
        Tensor tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
        Tensor tCgC_s2g = s2g_thr_copy_c.partition_D(gC);
        //
        // PREDICATES
        //

        Tensor tCpC_s2g = make_tensor<bool>(make_shape(size<1>(tCgC_s2g), size<2>(tCgC_s2g)),
                                        Stride<_1, Int<size<1>(tCgC_s2g)>>{}); // (CCPY_M, CCPY_N)
        Tensor tCcC_s2g = s2g_thr_copy_c.partition_S(cC);                          // (CCPY,CCPY_M,CCPY_N) -> (blk_m,blk_n)

#pragma unroll
        for (int m = 0; m < size<0>(tCpC_s2g); ++m) {
#pragma unroll
            for (int n = 0; n < size<1>(tCpC_s2g); ++n) {
                  tCpC_s2g(m, n) = elem_less(tCcC_s2g(0, m, n), make_coord(m_max_coord, n_max_coord));
            }
        }

        //
        // END PREDICATES
        //

        copy_if(s2g_tiled_copy_c, tCpC_s2g, tCsC_s2g, tCgC_s2g);
        //copy(s2g_tiled_copy_c, tCsC_s2g, tCgC_s2g);
    }else {

        auto t = make_tensor_like<OutType>(tCrC);
        copy(tCrC, t); // Convert precision

        typename Spec::TiledCopyO_R2S r2s_tiled_copy_o;
        ThrCopy r2s_thr_copy_o = r2s_tiled_copy_o.get_slice(tid);
        Tensor tOrC_r2s = r2s_thr_copy_o.retile_S(t);     // (CPY, CPY_M, CPY_N)
        Tensor tOsO_r2s = r2s_thr_copy_o.partition_D(s0); // (CPY, CPY_M, CPY_N)
        copy(r2s_tiled_copy_o, tOrC_r2s, tOsO_r2s);

        __syncthreads();

        typename Spec::TiledCopyO_S2G s2g_tiled_copy_o;
        ThrCopy s2g_thr_copy_o = s2g_tiled_copy_o.get_slice(tid);
        Tensor tOsO_s2g = s2g_thr_copy_o.partition_S(s0); // (CPY, CPY_M, CPY_N)
        Tensor tOgO_s2g = s2g_thr_copy_o.partition_D(g0); // (CPY, CPY_M, CPY_N)

        //
        // PREDICATES
        //

        Tensor tOpO_s2g = make_tensor<bool>(make_shape(size<1>(tOgO_s2g), size<2>(tOgO_s2g)),
                                            Stride<_1, Int<size<1>(tOgO_s2g)>>{}); // (OCPY_M, OCPY_N)
        Tensor cO = make_identity_tensor(make_shape(size<0>(s0), size<1>(s0)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tOcO_s2g = s2g_thr_copy_o.partition_S(cO);                          // (OCPY,OCPY_M,OCPY_N) -> (blk_m,blk_n)

#pragma unroll
        for (int m = 0; m < size<0>(tOpO_s2g); ++m) {
#pragma unroll
            for (int n = 0; n < size<1>(tOpO_s2g); ++n) {
                tOpO_s2g(m, n) = elem_less(tOcO_s2g(0, m, n), make_coord(m_max_coord, n_max_coord));
            }
        }

        //
        // END PREDICATES
        //

        copy_if(s2g_tiled_copy_o, tOpO_s2g, tOsO_s2g, tOgO_s2g);
        //copy(s2g_tiled_copy_o, tOsO_s2g, tOgO_s2g);
    }
}

template <typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ __launch_bounds__(Spec::kThreadNum) void hgemm_cute(void *__restrict__ Cptr,
                                                               const void *__restrict__ Aptr,
                                                               const void *__restrict__ Bptr,
                                                               int M,
                                                               int N,
                                                               int K,
                                                               void *__restrict__ Outptr) {
  using namespace cute;

  using X = Underscore;
  using MMA_shape = typename Spec::MMA_shape;
  using OutType = typename Spec::OutType;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using SmemLayoutA = typename Spec::SmemLayoutA;
  using SmemLayoutB = typename Spec::SmemLayoutB;
  using SmemLayoutC = typename Spec::SmemLayoutC;
  using SmemLayoutO = typename Spec::SmemLayoutO;

  constexpr int kBlockM = Spec::kBlockM;
  constexpr int kBlockN = Spec::kBlockN;
  constexpr int kBlockK = Spec::kBlockK;
  constexpr int kShmSizeA = Spec::kShmSizeA;
  constexpr int kShmSizeB = Spec::kShmSizeB;
  constexpr int G2S_Stages = Spec::G2S_Stages;
  constexpr int kMmaValExpandK = Spec::kMmaValExpandK;

  extern __shared__ __align__(1024) uint8_t smem[];

  uint8_t *Aptr_smem = smem;
  uint8_t *Bptr_smem = smem + kShmSizeA;
  uint8_t *Cptr_smem;
  if constexpr (!IsGemm)
    Cptr_smem = smem + kShmSizeA + kShmSizeB;
  else
    Cptr_smem = smem;
  uint8_t *Optr_smem = smem;

  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;

  Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA *)Aptr), make_shape(M, K), make_stride(K, Int<1>{})); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB *)Bptr), make_shape(N, K), make_stride(K, Int<1>{})); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC *)Cptr), make_shape(M, N), make_stride(N, Int<1>{})); // (M, N)
  Tensor mO = make_tensor(make_gmem_ptr((OutType *)Outptr), make_shape(M, N), make_stride(N, Int<1>{}));    // (M, N)

  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto coord = make_coord(bidy, bidx, _);

  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, K_TILES)
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, K_TILES)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)
  Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

  auto m_max_coord = M - size<0>(gA) * bidy;      // M - BLK_M * m_coord
  auto n_max_coord = N - size<0>(gB) * bidx;      // N - BLK_N * n_coord
  auto k_residue = K - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

  gA = domain_offset(make_coord(0, k_residue, 0), gA);
  gB = domain_offset(make_coord(0, k_residue, 0), gB);

  Tensor sA = make_tensor(make_smem_ptr((ComputeTypeA *)Aptr_smem), SmemLayoutA{}); // (BLK_M, BLK_K, G2S_PIPE)
  Tensor sB = make_tensor(make_smem_ptr((ComputeTypeB *)Bptr_smem), SmemLayoutB{}); // (BLK_N, BLK_K, G2S_PIPE)
  Tensor sC = make_tensor(make_smem_ptr((ComputeTypeC *)Cptr_smem), SmemLayoutC{}); // (BLK_M, BLK_N)
  Tensor sO = make_tensor(make_smem_ptr((OutType *)Optr_smem), SmemLayoutO{});      // (BLK_M, BLK_N)

  typename Spec::TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  Tensor tCrC = thr_mma.partition_fragment_C(gC);          // (MMA, MMA_M, MMA_N)

  typename Spec::TiledCopyA_G2S g2s_tiled_copy_a;
  ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
  Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA); // (ACPY, ACPY_M, ACPY_K, K_TILES)
  Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA); // (ACPY, ACPY_M, ACPY_K, G2S_PIPE)

  typename Spec::TiledCopyB_G2S g2s_tiled_copy_b;
  ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
  Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB); // (BCPY, BCPY_N, BCPY_K, K_TILES)
  Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB); // (BCPY, BCPY_N, BCPY_K, G2S_PIPE)

  typename Spec::TiledCopyC_G2S g2s_tiled_copy_c;
  ThrCopy g2s_thr_copy_c = g2s_tiled_copy_c.get_slice(tid);
  Tensor tCgC_g2s = g2s_thr_copy_c.partition_S(gC); // (CCPY, CCPY_M, CCPY_N)
  Tensor tCsC_g2s = g2s_thr_copy_c.partition_D(sC); // (CCPY, CCPY_M, CCPY_N)

  //
  // PREDICATES
  //

  Tensor tApA_g2s =
      make_tensor<bool>(make_shape(size<1>(tAsA_g2s), size<2>(tAsA_g2s)), Stride<_1, _0>{}); // (ACPY_M, ACPY_K)
  Tensor tBpB_g2s =
      make_tensor<bool>(make_shape(size<1>(tBsB_g2s), size<2>(tBsB_g2s)), Stride<_1, _0>{}); // (BCPY_N, BCPY_K)
  Tensor tCpC_g2s = make_tensor<bool>(make_shape(size<1>(tCsC_g2s), size<2>(tCsC_g2s)),
                                      Stride<_1, Int<size<1>(tCsC_g2s)>>{}); // (CCPY_M, CCPY_N)

  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (BLK_N,BLK_K) -> (blk_n,blk_k)
  Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (BLK_M,BLK_N) -> (blk_m,blk_n)

  Tensor tAcA_g2s = g2s_thr_copy_a.partition_S(cA); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tBcB_g2s = g2s_thr_copy_b.partition_S(cB); // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
  Tensor tCcC_g2s = g2s_thr_copy_c.partition_S(cC); // (CCPY,CCPY_M,CCPY_N) -> (blk_m,blk_n)

#pragma unroll
  for (int m = 0; m < size<0>(tApA_g2s); ++m) {
    tApA_g2s(m, 0) = get<0>(tAcA_g2s(0, m, 0)) < m_max_coord;
  }
#pragma unroll
  for (int n = 0; n < size<0>(tBpB_g2s); ++n) {
    tBpB_g2s(n, 0) = get<0>(tBcB_g2s(0, n, 0)) < n_max_coord;
  }
#pragma unroll
  for (int m = 0; m < size<0>(tCpC_g2s); ++m) {
#pragma unroll
    for (int n = 0; n < size<1>(tCpC_g2s); ++n) {
      tCpC_g2s(m, n) = elem_less(tCcC_g2s(0, m, n), make_coord(m_max_coord, n_max_coord));
    }
  }

  //
  // END PREDICATES
  //

  typename Spec::TiledCopyA_S2R s2r_tiled_copy_a;
  ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
  Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K, PIPE)
  Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyB_S2R s2r_tiled_copy_b;
  ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_M, CPY_K, PIPE)
  Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyC_S2R s2r_tiled_copy_c;
  ThrCopy s2r_thr_copy_c = s2r_tiled_copy_c.get_slice(tid);
  Tensor tCsC_s2r = s2r_thr_copy_c.partition_S(sC); // (CPY, CPY_M, CPY_K)
  Tensor tCrC_s2r = s2r_thr_copy_c.retile_D(tCrC);  // (CPY, CPY_M, CPY_K)

  //
  // Prefetch
  //

  // Interior fast path: block fully inside both M and N tiles and K is
  // tile-aligned. Used for stage 0 only (later prologue stages + mainloop
  // keep their existing predicated path with overshoot guards).
  const bool is_interior = (m_max_coord >= kBlockM) && (n_max_coord >= kBlockN) && (k_residue == 0);

  if constexpr (!IsGemm) {
    if (is_interior) {
      copy(g2s_tiled_copy_c, tCgC_g2s, tCsC_g2s);
    } else {
      clear(tCsC_g2s);
      copy_if(g2s_tiled_copy_c, tCpC_g2s, tCgC_g2s, tCsC_g2s);
    }
  }

  int NTilesK = ceil_div(K, kBlockK);

  if (is_interior) {
    copy(g2s_tiled_copy_a, tAgA_g2s(_, _, _, 0), tAsA_g2s(_, _, _, 0));
    copy(g2s_tiled_copy_b, tBgB_g2s(_, _, _, 0), tBsB_g2s(_, _, _, 0));
  } else {
    clear(tAsA_g2s);
    clear(tBsB_g2s);

#pragma unroll
    for (int k = 0; k < size<2>(tAsA_g2s); ++k) {
      if (get<1>(tAcA_g2s(0, 0, k)) >= -k_residue) {
        copy_if(g2s_tiled_copy_a, tApA_g2s(_, k), tAgA_g2s(_, _, k, 0), tAsA_g2s(_, _, k, 0));
      }
    }

#pragma unroll
    for (int k = 0; k < size<2>(tBsB_g2s); ++k) {
      if (get<1>(tBcB_g2s(0, 0, k)) >= -k_residue) {
        copy_if(g2s_tiled_copy_b, tBpB_g2s(_, k), tBgB_g2s(_, _, k, 0), tBsB_g2s(_, _, k, 0));
      }
    }
  }

  cp_async_fence();

#pragma unroll
  for (int ik = 1; ik < G2S_Stages - 1; ++ik) {
    // Set all predicates to false if we are going to overshoot bounds
    if (ik == NTilesK) {
      clear(tApA_g2s);
      clear(tBpB_g2s);
    }

    copy_if(g2s_tiled_copy_a, tApA_g2s, tAgA_g2s(_, _, _, ik), tAsA_g2s(_, _, _, ik));
    copy_if(g2s_tiled_copy_b, tBpB_g2s, tBgB_g2s(_, _, _, ik), tBsB_g2s(_, _, _, ik));

    cp_async_fence();
  }

  // Prefetch register stage

  constexpr int k_tiles = size<2>(tArA_s2r);
  constexpr int prefetch_s2r_tiles = 1;

  cp_async_wait<G2S_Stages - 2>();
  __syncthreads();

#pragma unroll
  for (int k = 0; k < prefetch_s2r_tiles; ++k) {
    copy(s2r_tiled_copy_a, tAsA_s2r(_, _, k, 0), tArA_s2r(_, _, k));
    copy(s2r_tiled_copy_b, tBsB_s2r(_, _, k, 0), tBrB_s2r(_, _, k));
  }

  if constexpr (IsGemm) {
    clear(tCrC); // Set the accumulators to zero
  } else {
    copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r);
  }

  //
  // MAINLOOP
  //

  int g2s_gmem_pipe = G2S_Stages - 1;
  int g2s_smem_pipe = G2S_Stages - 1;
  int s2r_smem_pipe = 0;

  for (int ik = 0; ik < NTilesK; ++ik) {
    // Note, the for_each() function is required here to ensure `k` is of type Int<N>.
    for_each(make_int_sequence<k_tiles>{}, [&](auto k) {
      if (k == k_tiles - prefetch_s2r_tiles) {
        cp_async_wait<G2S_Stages - 2>();
        __syncthreads();
        ++s2r_smem_pipe;
        s2r_smem_pipe = (s2r_smem_pipe == G2S_Stages) ? 0 : s2r_smem_pipe;
      }

      // copy A and B
      auto k_next = (k + Int<prefetch_s2r_tiles>{}) % k_tiles;
      copy(s2r_tiled_copy_a, tAsA_s2r(_, _, k_next, s2r_smem_pipe), tArA_s2r(_, _, k_next));
      copy(s2r_tiled_copy_b, tBsB_s2r(_, _, k_next, s2r_smem_pipe), tBrB_s2r(_, _, k_next));

      if (k == 0) {
        // Set all predicates to false if we are going to overshoot bounds
        if (g2s_gmem_pipe == NTilesK) {
          clear(tApA_g2s);
          clear(tBpB_g2s);
        }

        copy_if(g2s_tiled_copy_a, tApA_g2s, tAgA_g2s(_, _, _, g2s_gmem_pipe), tAsA_g2s(_, _, _, g2s_smem_pipe));
        copy_if(g2s_tiled_copy_b, tBpB_g2s, tBgB_g2s(_, _, _, g2s_gmem_pipe), tBsB_g2s(_, _, _, g2s_smem_pipe));

        cp_async_fence();
        ++g2s_gmem_pipe;
        ++g2s_smem_pipe;
        g2s_smem_pipe = (g2s_smem_pipe == G2S_Stages) ? 0 : g2s_smem_pipe;
      }

#pragma unroll
      for (int ik = k * kMmaValExpandK; ik < (k + 1) * kMmaValExpandK; ++ik) {
        gemm(tiled_mma, tCrC, tCrA(_, _, ik), tCrB(_, _, ik), tCrC);
      }
    });
  }

  cp_async_wait<0>();
  __syncthreads();

  if constexpr (!IsCvtPrecision) {
    typename Spec::TiledCopyC_R2S r2s_tiled_copy_c;
    ThrCopy r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(tid);
    Tensor tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);  // (CPY, CPY_M, CPY_N)
    Tensor tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);

    __syncthreads();

    typename Spec::TiledCopyC_S2G s2g_tiled_copy_c;
    ThrCopy s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(tid);
    Tensor tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, CPY_M, CPY_N)
    Tensor tCgC_s2g = s2g_thr_copy_c.partition_D(gC); // (CPY, CPY_M, CPY_N)

    //
    // PREDICATES
    //

    Tensor tCpC_s2g = make_tensor<bool>(make_shape(size<1>(tCgC_s2g), size<2>(tCgC_s2g)),
                                        Stride<_1, Int<size<1>(tCgC_s2g)>>{}); // (CCPY_M, CCPY_N)
    Tensor tCcC_s2g = s2g_thr_copy_c.partition_S(cC);                          // (CCPY,CCPY_M,CCPY_N) -> (blk_m,blk_n)

#pragma unroll
    for (int m = 0; m < size<0>(tCpC_s2g); ++m) {
#pragma unroll
      for (int n = 0; n < size<1>(tCpC_s2g); ++n) {
        tCpC_s2g(m, n) = elem_less(tCcC_s2g(0, m, n), make_coord(m_max_coord, n_max_coord));
      }
    }

    //
    // END PREDICATES
    //

    copy_if(s2g_tiled_copy_c, tCpC_s2g, tCsC_s2g, tCgC_s2g);

  } else {

    auto t = make_tensor_like<OutType>(tCrC);
    copy(tCrC, t); // Convert precision

    typename Spec::TiledCopyO_R2S r2s_tiled_copy_o;
    ThrCopy r2s_thr_copy_o = r2s_tiled_copy_o.get_slice(tid);
    Tensor tOrC_r2s = r2s_thr_copy_o.retile_S(t);     // (CPY, CPY_M, CPY_N)
    Tensor tOsO_r2s = r2s_thr_copy_o.partition_D(sO); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_o, tOrC_r2s, tOsO_r2s);

    __syncthreads();

    typename Spec::TiledCopyO_S2G s2g_tiled_copy_o;
    ThrCopy s2g_thr_copy_o = s2g_tiled_copy_o.get_slice(tid);
    Tensor tOsO_s2g = s2g_thr_copy_o.partition_S(sO); // (CPY, CPY_M, CPY_N)
    Tensor tOgO_s2g = s2g_thr_copy_o.partition_D(gO); // (CPY, CPY_M, CPY_N)

    //
    // PREDICATES
    //

    Tensor tOpO_s2g = make_tensor<bool>(make_shape(size<1>(tOgO_s2g), size<2>(tOgO_s2g)),
                                        Stride<_1, Int<size<1>(tOgO_s2g)>>{}); // (OCPY_M, OCPY_N)
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tOcO_s2g = s2g_thr_copy_o.partition_S(cO);                          // (OCPY,OCPY_M,OCPY_N) -> (blk_m,blk_n)

#pragma unroll
    for (int m = 0; m < size<0>(tOpO_s2g); ++m) {
#pragma unroll
      for (int n = 0; n < size<1>(tOpO_s2g); ++n) {
        tOpO_s2g(m, n) = elem_less(tOcO_s2g(0, m, n), make_coord(m_max_coord, n_max_coord));
      }
    }

    //
    // END PREDICATES
    //

    copy_if(s2g_tiled_copy_o, tOpO_s2g, tOsO_s2g, tOgO_s2g);
  }
}
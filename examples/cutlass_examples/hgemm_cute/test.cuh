#include "cute/pointer.hpp"
#include "cute/tensor_impl.hpp"
#include <__clang_cuda_runtime_wrapper.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>


namespace Spec {
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

            using MMA_op = std::conditional_t<std::is_same_v<ComputeTypeA, cute::bfloat16_t> && std::is_same_v<ComputeTypeB, cute::bfloat16_t> 
                                                && std::is_same_v<ComputeTypeC, float>, SM80_16x8x16_F32BF16BF16F32_TN,
                                              std::conditional_t<std::is_same_v<ComputeTypeA, cute::half_t> && std::is_same_v<ComputeTypeB, cute::half_t> && std::is_same_v<ComputeTypeC, cute::half_t>, SM80_16x8x16_F16F16F16F16_TN,
                                              std::conditional_t<std::is_same_v<ComputeTypeA, cute::half_t> && std::is_same_v<ComputeTypeB, cute::half_t> && std::is_same_v<ComputeTypeC, float>, SM80_16x8x16_F32F16F16F32_TN, void>>>;
            static_assert(!std::is_same_v<MMA_op, void>, "Unsupported MMA op!"); 
            
            using MMA_traits = MMA_Traits<MMA_op>;
            using MMA_Atom = MMA_Atom<MMA_traits>;
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

            using MMAThrLayout = decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
            using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
            using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

            using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
            using Copy_S2R_op_A = std::conditional_t<sizeof(ComputeTypeA) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
            using Copy_S2R_op_B = std::conditional_t<sizeof(ComputeTypeB) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
            using Copy_S2R_op_C = std::conditional_t<sizeof(ComputeTypeB) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;

            using CopyA_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeA>;
            using CopyB_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeB>;
            using CopyC_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeC>;

            using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op_A, ComputeTypeA>;
            using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op_B, ComputeTypeB>;
            using CopyC_S2R_atom = Copy_Atom<Copy_S2R_op_C, ComputeTypeC>;

        #if (defined (__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            using Copy_R2S_op = SM90_U32x2_STSM_N;
        #else 
            using CopyR2S_op = AutoVectorizingCopy;
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
                                            make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}), make_stride(Int<kBlockK_Copy>{}, Int<1>{})),
                                            make_layout(make_shape(Int<1>{}, Int<8>{}))));
            
            using TiledCopyA_S2R = decltype(make_tiled_copy(CopyA_S2R_atom{}, TiledMMA{}));
            using TiledCopyB_S2R = decltype(make_tiled_copy(CopyB_S2R_atom{}, TiledMMA{}));
            using TiledCopyC_S2R = decltype(make_tiled_copy(CopyC_S2R_atom{}, TiledMMA{}));

            using TiledCopyC_R2S = decltype(make_tiled_copy(CopyC_R2S_atom{}, TiledMMA{}));
            using TiledCopyO_R2S = decltype(make_tiled_copy(CopyC_R2S_atom{}, TiledMMA{}));

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
                                                            make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockK)>{}), 
                                                            make_stride(Int<cute::min(64, kBlockK)>{}, Int<1>{}))));
            using SmemLayoutAtomO = decltype(composition(Swizzle<3, 3, 3>{},
                                                            make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockK)>{}), 
                                                            make_stride(Int<cute::min(64, kBlockK)>{}, Int<1>{}))));
         
            using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_layout(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{})));
            using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_layout(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{})));
            using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{}, make_layout(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{})));
            using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, make_layout(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{})));

            static constexpr int kShmSizeA = cosize(SmemLayoutA{}) * sizeof(ComputeTypeA);
            static constexpr int kShmSizeB = cosize(SmemLayoutB{}) * sizeof(ComputeTypeB);
            static constexpr int kShmSizeC = cosize(SmemLayoutC{}) * sizeof(ComputeTypeC);
            static constexpr int kShmSizeO = cosize(SmemLayoutO{}) * sizeof(OutType);
        };
}

template<typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ __launch_bounds__(Spec::kThreadNum) void hgemm_cute(void *__restrict__ Cptr, const void *__restrict__ Aptr, const void *__restrict__ Bptr, int m, int n, int k, const void *__restrict__ Outptr) {

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

    constexpr int kBlockM = Spec::kBLockM;
    constexpr int kBlockN = Spec::kBlockN;
    constexpr int kBlockK = Spec::kBlockK;
    constexpr int kShmSizeA = Spec::kShmSizeA;
    constexpr int kShmSizeB = Spec::kShmSizeB;
    constexpr int G2S_Stages = Spec::G2S_Stages;

    extern __shared__ __align__(1024) uint8_t smem[]; 
    
    uint8_t *Aptr_smem = smem;
    uint8_t *Bptr_smem = smem + kShmSizeA;
    uint8_t *Cptr_smem;
    if constexpr (!IsGemm){
        Cptr_smem = smem + kShmSizeA + kShmSizeB;
    }else {
        Cptr_smem = smem;
    }
    uint8_t *Optr_smem = smem;

    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA*)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB*)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC*)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
    Tensor m0 = make_tensor(make_gmem_ptr((OutType*)Outptr), make_shape(m, n), make_stride(n, Int<1>{}));

    auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
    auto coord = make_coord(bidy, bidx, _);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); 
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});
    Tensor g0 = local_tile(m0, tiler, coord, Step<_1, _1,X>{});

    auto m_max_coord = m - size<0>(gA) * bidy;
    auto n_max_coord = n - size<0>(gB) * bidx;
    auto k_residue = k - size<1>(gA) * size<2>(gA);

    gA = domain_offset(make_coord(0, k_residue, 0), gA);
    gB = domian_offset(make_coord(0, k_residue, 0), gB);

    Tensor sA = make_tensor(make_smem_ptr((ComputeTypeA*)Aptr_smem), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr((ComputeTypeB*)Bptr_smem), SmemLayoutB{});
    Tensor sC = make_tensor(make_smem_ptr((ComputeTypeC*)Cptr_smem), SmemLayoutC{});
    Tensor sO = make_tensor(make_smem_ptr((OutType*)Optr_smem), SmemLayoutO{});

    typename Spec::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    Tensor tCrC = thr_mma.partition_fragment_C(gC);

    typename Spec::TiledCopyA_G2S g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA);
    Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA);

    typename Spec::TiledCopyB_G2S g2s_tiled_copy_b;
    ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);    
    Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB);
    Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB);

    typename Spec::TiledCopyC_G2S g2s_tiled_copy_c;
    ThrCopy g2s_thr_copy_c = g2s_tiled_copy_c.get_slice(tid);
    Tensor tCgC_g2s = g2s_thr_copy_c.partition_S(gC);
    Tensor tCsC_g2s = g2s_thr_copy_c.partition_D(sC);

    Tensor tApA_g2s = make_tensor<bool>(make_shape(size<1>(tAsA_g2s), size<2>(tAsA_g2s)));
    Tensor tBpB_g2s = make_tensor<bool>(make_shape(size<1>(tBsB_g2s), size<2>(tBsB_g2s)));
    Tensor tCpC_g2s = make_tensor<bool>(make_shape(size<1>(tCsC_g2s), size<2>(tCsC_g2s)));

    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));
    Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC)));

    Tensor tAcA_g2s = g2s_thr_copy_a.partition_S(cA);
    Tensor tBcB_g2s = g2s_thr_copy_a.partittion_S(cB);
    Tensor tCcC_g2s = g2s_thr_copy_c.partition_S(cC);

#pragma unroll
    for(int m = 0; m < size<0>(tApA_g2s); ++m) {
        tApA_g2s(m, 0) = get<0>(tAcA_g2s(0, m, 0)) < m_max_coord;
    }

#pragma unroll
    for(int n = 0; n < size<0>(tBpB_g2s); ++n) {
        tBpB_g2s(n, 0) = get<0>(tAcA_g2s(0, n, 0)) < m_max_coord; 
    }

#pragma unroll
    for(int m = 0; m < size<0>(tCpC_g2s); ++m) {
#pragma unroll 
        for(int n = 0; n < size<1>(tCpC_g2s); ++n) {
            tCpC_g2s(m, n) = elem_less(tCcC_g2s(0, m, n), make_coord(m_max_coord, n_max_coord));
        }
    }
    
    typename Spec::TiledCopyA_S2R s2r_tiled_copy_a;
    ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA);
    Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);

    typename Spec::TiledCopyB_S2R s2r_tiled_copy_b;
    ThrCopy s2r_thr_copy_b = s2r_tiled_copy_a.get_slice(tid);
    Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB);
    Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);

    typename Spec::TiledCopyC_S2R s2r_tiled_copy_c;
    ThrCopy s2r_thr_copy_c = s2r_tiled_copy_c.get_slice(tid);  
    Tensor tCsC_s2r = s2r_thr_copy_c.partition_S(sC);
    Tensor tCrC_sr2 = s2r_thr_copy_c.retile_D(tCrC);

    if constexpr (!IsGemm) {
        clear(tCsC_g2s);
        copy_if(g2s_tiled_copy_c, tCpC_g2s, tCgC_g2s, tCsC_g2s);
    } 

    int NTilesK = ceil_div(k, kBlockK);

    clear(tAsA_g2s);
    clear(tBsB_g2s);

#pragma unroll
    for(int k = 0; k < size<2>(tAsA_g2s); ++k) {
        if(get<1>(tAcA_g2s(0, 0, k)) >= -k_residue) {
            copy_if(g2s_tiled_copy_a, tApA_g2s(_, k), tAgA_g2s(_, _, k, 0), tAsA_g2s(_, _, k, 0));
        }
    }

#pragma unroll
    for(int k = 0; k < size<2>(tBsB_g2s); ++k) {
        if(get<1>(tBcB_g2s(0, 0, k)) >= -k_residue) {
            copy_if(g2s_tiled_copy_b, tBpB_g2s(_, k), tBgB_g2s(_, _, k, 0), tBsB_g2s(_, _, k, 0));
        }
    }

    cp_async_fence();


#pragma unroll
    for(int ik = 1; ik < G2S_Stages - 1; ++ik) {

        if(ik == NTilesK) {
            clear(tApA_g2s);    
            clear(tBpB_g2s);
        }

        copy_if(g2s_tiled_copy_a, tApA_g2s, tAgA_g2s(_, _, _, ik), tAsA_g2s(_, _, _, ik));
        copy_if(g2s_tiled_copy_b, tBpB_g2s, tBgB_g2s(_, _, _, ik), tBsB_g2s(_, _, _, ik));

        cp_async_fence();
    }

    cp_async_wait<G2S_Stages - 2>();
    __syncthreads();

    int g2s_gmem_pipe = G2S_Stages - 1;
    int g2s_smem_pipe = G2S_Stages - 1;
    int s2r_smem_pipe = 0;

    for(int ik = 0; ik < NTilesK; ik ++) {
        
        copy(s2r_tiled_copy_a, tAsA_s2r(_, _, _, s2r_smem_pipe), tArA_s2r);
        copy(s2r_tiled_copy_b, ); 

        {

        }
        if(ik == 0) {

        }
    }

}

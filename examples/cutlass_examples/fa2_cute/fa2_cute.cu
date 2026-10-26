#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/config.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "fa2.h"
#include <cmath>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace flash{
using namespace cute;
template<typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> &D) {

    CUTE_STATIC_ASSERT(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT(size<0>(S) == size<0>(D));  // MMA
    CUTE_STATIC_ASSERT(size<1>(S) == size<1>(D));  // MMA_M
    CUTE_STATIC_ASSERT(size<2>(S) == size<2>(D));  // MMA_K

    #pragma unroll
    for(int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for(int k = 0; k < size<2>(S); ++k) {
            cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        }
    }
}

template<typename Tensor0, typename Tensor1, 
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                                    Tensor4 const& tCsB, TiledMma tiled_mma, 
                                    TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                                    ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {

        CUTE_STATIC_ASSERT(size<1>(tCrA) == size<1>(acc));  // MMA_M
        CUTE_STATIC_ASSERT(size<1>(tCrB) == size<2>(acc));  // MMA_N
        CUDA_STATIC_ASSERT(size<2>(tCrA) == size<2>(tCrB));  // MMA_K
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA); 
        CUTE_STATIC_ASSERT(size<1>(tCsA) == size<1>(tCrA_copy_view));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT(size<1>(tCsB) == size<1>(tCrB_copy_view));
        
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        #pragma unroll
        for(int i = 0; i < size<2>(tCrA); i++) {
            if(i < size<2>(tCrA) - 1) {
                cute::copy(smem_tiled_copy_A, tCsA(), tCrA_copy_view());
                cute::copy(smem_tiled_copy_B, tCsB(), tCrB_copy_view());
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }

}

}

// Shared Storage with Aligned address
template<class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage{
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

void set_params_fprop(Flash_fwd_params &params,
                        const torch::Tensor q,
                        const torch::Tensor k,
                        const torch::Tensor v,
                        torch::Tensor out,
                        void *softmax_lse_d,
                        float softmax_scale,
                        bool is_causal) {

        memset(&params, 0, sizeof(params));

        params.bs = q.size(0);
        params.head = q.size(1);                    
        params.q_seqlen = q.size(2);
        params.dim = q.size(3);

        params.k_head = k.size(1);
        params.k_seqlen = k.size(2);

        params.bs_stride = q.stride(0);
        params.head_stride = q.stride(1);
        params.seqlen_stride = q.stride(2);
        params.dim_stride = q.stride(3);

        params.softmax_scale = softmax_scale;
        params.softmax_scale_log2 = softmax_scale * M_LOG2E;
        params.is_causal = is_causal; 
        params.is_bf16 = q.dtype() == torch::kBFloat16;

        params.softmax_lse_ptr = softmax_lse_d;

        params.q_ptr = q.data_ptr();
        params.k_ptr = k.data_ptr();
        params.v_ptr = v.data_ptr();
        params.out_ptr = out.data_ptr();
        return;
}

template<typename Kernel_traits, bool Is_causal = false, typename Params>
__global__ void flash_attention2_cute(const Params params) {

    using namespace cute;

    const int m_block = blockIdx.x;

    const int base_id = blockIdx.y;
    const int tidx = threadIdx.x;

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using TiledMMA = typename Kernel_traits::TiledMma;
    using index_t = typename Kernel_traits::index_t;
    using SmemLayoutQ = typename  Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename  Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename  Kernel_traits::SmemLayoutKV;
    using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockM = Kernel_traits::kBLockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    extern __shared__ char smem_[];
    using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;   
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);
    // (B, H, S, hd)
    const int bs_head_offset = base_id * params.head_stride;  // head_stride = S * hd

    Tensor Q = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + bs_head_offset),
                                        make_shape(params.q_seqlen, Int<kHeadDim>{}),
                                        make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor K = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset), 
                                        make_shape(params.k_seqlen, Int<kHeadDim>{}),
                                        make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor V = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset), 
                                        make_shape(params.v_seqlen, Int<kHeadDim>{}),
                                        make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor O = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset), 
                                        make_shape(params.o_seqlen, Int<kHeadDim>{}),
                                        make_stride(Int<kHeadDim>{}, Int<1>{}));

    Tensor LSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + base_id * params.q_seqlen),
                                        make_shape(params.q_seqlen),
                                        make_stride(Int<1>{}));    

    Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

    Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));


    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
    Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

    // gmem->smem
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    // smem->rmem
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma); 
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    
    flash::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cute::cp_async_fence();

    Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{}); 

    const int n_block_min = 0;
    int seqlen_start = m_block * kBlockM;
    int seqlen_end = (m_block + 1) * kBlockM;
    int n_block_max = Is_causal ? cute::ceil_div(seqlen_end, kBlockN) : cute::ceil_div(params.k_seqlen, kBlockN);

    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);

    clear(rAccOut);

    for(int nbi = n_block_min; nbi < n_block_max; nbi++) {
        
        auto rAccScore = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
        clear(rAccScore);


        flash::cp_async_wait<0>();
        __syncthreads();

        gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0)); 
        flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        cute::cp_async_fence();
        
        flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                            smem_thr_copy_Q, smem_thr_copy_K);
        Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore));

    }
}
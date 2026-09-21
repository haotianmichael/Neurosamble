// ===========================================================================
// Flash Attention v2 forward — CuTe C++ port for NVIDIA Blackwell GeForce SM120
// (RTX 5090, GB202).  Hand-translated from the CuTe DSL reference
// (FlashAttentionForwardSm120, the CpAsync variant).
//
// SM120 notes (why this is an Ampere-FA2 kernel, not an SM100/tcgen05 one):
//   * SM80-era `mma.sync.aligned.m16n8k16` tensor cores  -> SM80_16x8x16_*_TN
//   * cp.async for GMEM->SMEM                             -> SM80_CP_ASYNC_CACHEGLOBAL
//   * ldmatrix for SMEM->RMEM                             -> SM75_U32x4_LDSM_N / U16x8_LDSM_T
//   * NO tcgen05 / TMEM / wgmma  (SM100 datacenter only)
//   * ~99 KB smem/SM  -> single-buffered K, single-buffered V, one Q tile
//
// Build (needs CUTLASS 4.x headers + CUDA 12.8/13.x):
//   nvcc -std=c++17 -arch=sm_120a -O3 --expt-relaxed-constexpr \
//        -I/path/to/cutlass/include \
//        flash_fwd_sm120.cu -o flash_fwd_sm120
//
// NOTE: this is a careful source-level translation cross-checked against the
// DSL and the canonical CUTLASS/Tri-Dao FA2 C++ structure. It has NOT been
// compiled on SM120 hardware here (no GPU/toolchain in this environment).
// The built-in main() runs an fp32 CPU reference check so you can validate
// immediately on your 5090.
// ===========================================================================

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>

#include <cutlass/numeric_types.h>
#include <cutlass/cutlass.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <type_traits>

using namespace cute;

// ---------------------------------------------------------------------------
// Small device math helpers matching the DSL's fastmath ex2 / rcp.approx and
// the 4-thread ("thread quad") row reduction over the SM80 MMA C fragment.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float flash_exp2(float x) {
    float y; asm volatile("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ float flash_rcp(float x) {
    float y; asm volatile("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
// In the m16n8 SM80 C layout each output row is held by a quad of lanes
// (lane, lane^1, lane^2, lane^3). Reduce with xor-2 then xor-1.
__device__ __forceinline__ float quad_allreduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}
__device__ __forceinline__ float quad_allreduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// ---------------------------------------------------------------------------
// Accumulator relayout helpers (standard FA2 idioms).
//   rowcol : ((2,2),MMA_M,MMA_N) -> ((2,MMA_M),(2,MMA_N))  (per-thread row/col)
//   Aregs  : ((2,2),MMA_M,MMA_N) -> ((2,2,2),MMA_M,MMA_N/2) (reuse P as MMA A)
// ---------------------------------------------------------------------------
template <class Layout>
__device__ __forceinline__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});          // ((2,2),MMA_M,MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<2>(l)));
}
template <class Layout>
__device__ __forceinline__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});    // ((2,2),MMA_M,(2,MMA_N/2))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

// ---------------------------------------------------------------------------
// Kernel traits — all compile-time layouts / atoms, mirroring the DSL __call__.
// ---------------------------------------------------------------------------
template <class Element_, int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Fwd_traits {
    using Element      = Element_;
    using ElementAccum = float;

    static constexpr int kHeadDim  = kHeadDim_;
    static constexpr int kBlockM   = kBlockM_;
    static constexpr int kBlockN   = kBlockN_;
    static constexpr int kNWarps   = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    // --- swizzled smem atom: Swizzle(B,3,3) over an (8, kBlockKSmem) tile ---
    static constexpr int kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32;
    static constexpr int kSwizzleB   = (kBlockKSmem == 64) ? 3 : 2;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzleB, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    // Transposed V view (d, n) for the P*V GEMM (LDSM.T source).
    using SmemLayoutVt = decltype(composition(
        SmemLayoutKV{},
        make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{})));

    // --- GMEM tiled copies: cp.async (128b) for QKV load, universal for O ---
    static constexpr int kGmemElemsPerLoad   = 128 / cutlass::sizeof_bits<Element>::value; // 8
    static constexpr int kGmemThreadsPerRow  = kBlockKSmem / kGmemElemsPerLoad;            // 8
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "");
    using GmemThrLayout = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                 Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemValLayout = Layout<Shape<_1, Int<kGmemElemsPerLoad>>>;

    using GmemTiledCopyQKV = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
        GmemThrLayout{}, GmemValLayout{}));
    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{},
        GmemThrLayout{}, GmemValLayout{}));

    // --- Tiled MMA: SM80 16x8x16 TN, kNWarps along M ---
    using MmaAtom = MMA_Atom<std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        SM80_16x8x16_F32F16F16F32_TN,
        SM80_16x8x16_F32BF16BF16F32_TN>>;
    using TiledMMA = decltype(make_tiled_mma(
        MmaAtom{},
        Layout<Shape<Int<kNWarps>, _1, _1>>{},        // atom layout (nwarps,1,1)
        Tile<Int<kNWarps * 16>, _16, _16>{}));        // permutation MNK

    // --- SMEM->RMEM ldmatrix atoms ---
    using SmemCopyAtom           = Copy_Atom<SM75_U32x4_LDSM_N, Element>;  // Q, K
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;  // V^T
    using SmemCopyAtomO          = Copy_Atom<UniversalCopy<Element>, Element>; // R2S for O

    // Shared storage: Q/K/V, each 128B-aligned so the swizzle maps banks cleanly.
    struct SharedStorage {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>,  128> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutKV>, 128> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutKV>, 128> smem_v;
    };
    static constexpr int kSmemSize = int(sizeof(SharedStorage));
};

// ---------------------------------------------------------------------------
// Runtime parameters (contiguous (b, s, h, d) tensors; strides in elements).
// ---------------------------------------------------------------------------
struct Flash_params {
    const void* __restrict__ q_ptr;
    const void* __restrict__ k_ptr;
    const void* __restrict__ v_ptr;
    void*       __restrict__ o_ptr;
    int b, h, d;
    int seqlen_q, seqlen_k;
    int64_t q_batch_stride, q_head_stride, q_row_stride;
    int64_t k_batch_stride, k_head_stride, k_row_stride;
    int64_t v_batch_stride, v_head_stride, v_row_stride;
    int64_t o_batch_stride, o_head_stride, o_row_stride;
    float scale_log2;   // softmax_scale * log2(e)
};

// ---------------------------------------------------------------------------
// The kernel.
// ---------------------------------------------------------------------------
template <class Traits, bool Is_causal>
__global__ void __launch_bounds__(Traits::kNThreads, 1)
flash_fwd_kernel(const Flash_params params) {
    using Element = typename Traits::Element;
    constexpr int kBlockM  = Traits::kBlockM;
    constexpr int kBlockN  = Traits::kBlockN;
    constexpr int kHeadDim = Traits::kHeadDim;

    const int tidx    = threadIdx.x;
    const int m_block = blockIdx.x;
    const int bidb    = blockIdx.y;   // batch
    const int bidh    = blockIdx.z;   // head

    // Number of K tiles this CTA visits (causal: stop at the diagonal).
    int n_block_max = cute::ceil_div(params.seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = min(n_block_max,
                          cute::ceil_div((m_block + 1) * kBlockM, kBlockN));
    }
    const int n_block_first = n_block_max - 1;  // highest index, processed first

    // ---- Global tensors for this (batch, head), sliced to (seq, d) ----
    Element const* q_gptr = reinterpret_cast<Element const*>(params.q_ptr)
                          + bidb * params.q_batch_stride + bidh * params.q_head_stride;
    Element const* k_gptr = reinterpret_cast<Element const*>(params.k_ptr)
                          + bidb * params.k_batch_stride + bidh * params.k_head_stride;
    Element const* v_gptr = reinterpret_cast<Element const*>(params.v_ptr)
                          + bidb * params.v_batch_stride + bidh * params.v_head_stride;
    Element*       o_gptr = reinterpret_cast<Element*>(params.o_ptr)
                          + bidb * params.o_batch_stride + bidh * params.o_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(q_gptr),
                            make_shape(params.seqlen_q, params.d),
                            make_stride(params.q_row_stride, _1{}));
    Tensor mK = make_tensor(make_gmem_ptr(k_gptr),
                            make_shape(params.seqlen_k, params.d),
                            make_stride(params.k_row_stride, _1{}));
    Tensor mV = make_tensor(make_gmem_ptr(v_gptr),
                            make_shape(params.seqlen_k, params.d),
                            make_stride(params.v_row_stride, _1{}));
    Tensor mO = make_tensor(make_gmem_ptr(o_gptr),
                            make_shape(params.seqlen_q, params.d),
                            make_stride(params.o_row_stride, _1{}));

    Tensor gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{})); // (M,K)
    Tensor gK = local_tile(mK, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));        // (N,K,nblk)
    Tensor gV = local_tile(mV, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));        // (N,K,nblk)
    Tensor gO = local_tile(mO, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));  // (M,K)

    // ---- Shared memory tensors ----
    extern __shared__ char smem_raw[];
    auto& shared = *reinterpret_cast<typename Traits::SharedStorage*>(smem_raw);
    Tensor sQ  = make_tensor(make_smem_ptr(shared.smem_q.data()), typename Traits::SmemLayoutQ{});
    Tensor sK  = make_tensor(make_smem_ptr(shared.smem_k.data()), typename Traits::SmemLayoutKV{});
    Tensor sV  = make_tensor(make_smem_ptr(shared.smem_v.data()), typename Traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(make_smem_ptr(shared.smem_v.data()), typename Traits::SmemLayoutVt{});

    // ---- GMEM tiled-copy partitions ----
    typename Traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);   // ((v),M,K)
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);   // ((v),M,K,nblk)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);   // ((v),M,K,nblk)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // ---- MMA partitions / accumulators ----
    typename Traits::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                  // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                  // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma.partition_fragment_B(sVt);                 // (MMA,MMA_N,MMA_K)
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{}); // (MMA,MMA_M,MMA_N)
    clear(acc_o);

    // ---- SMEM->RMEM ldmatrix tiled copies ----
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSrQ_view = smem_thr_copy_Q.retile_D(tSrQ);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tSrK_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tOrVt_view = smem_thr_copy_V.retile_D(tOrVt);

    // ---- Predication (identity coords over (seq, d)) ----
    Tensor cQ  = local_tile(make_identity_tensor(make_shape(params.seqlen_q, params.d)),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
    Tensor cKV = local_tile(make_identity_tensor(make_shape(params.seqlen_k, params.d)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block_first, _0{}));
    Tensor tQcQ  = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    // Per-(k-tile) head-dim predicates (a 128b load is wholly in/out since 8|d).
    Tensor tQpQ   = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tQsQ); ++k)  tQpQ(k)   = get<1>(tQcQ(_0{}, _0{}, k))   < params.d;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tKsK); ++k)  tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < params.d;

    // Predicated tile load helper: cp.async each in-bounds (m,k) vector, zero-fill OOB.
    auto load_tile = [&](auto tiled_copy, auto const& gS, auto& sD, auto const& cS,
                         auto const& pKd, int max_seq, auto row_guard) {
        CUTE_UNROLL
        for (int m = 0; m < size<1>(gS); ++m) {
            bool row_ok = (!decltype(row_guard)::value) || (get<0>(cS(_0{}, m, _0{})) < max_seq);
            if (row_ok) {
                CUTE_UNROLL
                for (int k = 0; k < size<2>(gS); ++k) {
                    if (pKd(k)) cute::copy(tiled_copy, gS(_, m, k), sD(_, m, k));
                    else        cute::clear(sD(_, m, k));
                }
            } else {
                cute::clear(sD(_, m, _));
            }
        }
    };

    // ---- Prologue: async-load Q and the first K tile ----
    load_tile(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, params.seqlen_q, cute::true_type{});
    load_tile(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block_first), tKsK, tKVcKV, tKVpKV,
              params.seqlen_k, cute::true_type{});
    cute::cp_async_fence();

    // ---- Softmax running state (one entry per row this thread owns) ----
    Tensor acc_o_rc = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    Tensor row_max  = make_tensor<float>(make_shape(size<0>(acc_o_rc)));
    Tensor row_sum  = make_tensor<float>(make_shape(size<0>(acc_o_rc)));
    CUTE_UNROLL
    for (int i = 0; i < size(row_max); ++i) { row_max(i) = -INFINITY; row_sum(i) = 0.f; }

    // -----------------------------------------------------------------------
    // Online-softmax over one K/V tile (is_first controls rescale/V-load path;
    // in_mask applies causal / seqlen_k masking).
    // -----------------------------------------------------------------------
    auto compute_one_n_block = [&](int n_block, auto is_first_c, auto in_mask_c) {
        constexpr bool Is_first = decltype(is_first_c)::value;
        constexpr bool In_mask  = decltype(in_mask_c)::value;

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); // (MMA,MMA_M,MMA_N)
        clear(acc_s);

        // Wait for this tile's K (and Q on the first iteration), then load V.
        cute::cp_async_wait<0>();
        __syncthreads();

        if (Is_first) {
            load_tile(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
                      params.seqlen_k, cute::true_type{});
        } else {
            load_tile(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
                      params.seqlen_k, cute::false_type{});
        }
        cute::cp_async_fence();

        // ---- S = Q * K^T  (ldmatrix-pipelined over MMA_K) ----
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, _0{}), tSrQ_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}), tSrK_view(_, _, _0{}));
        CUTE_UNROLL
        for (int k = 0; k < size<2>(tSsQ); ++k) {
            int k_next = (k + 1) % size<2>(tSsQ);
            cute::copy(smem_tiled_copy_Q, tSsQ(_, _, k_next), tSrQ_view(_, _, k_next));
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_next), tSrK_view(_, _, k_next));
            cute::gemm(tiled_mma, tSrQ(_, _, k), tSrK(_, _, k), acc_s);
        }

        // Wait for V, then prefetch the next K tile (overlaps softmax + PV).
        cute::cp_async_wait<0>();
        __syncthreads();
        if (n_block > 0) {
            load_tile(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV,
                      params.seqlen_k, cute::false_type{});
            cute::cp_async_fence();
        }

        // ---- Online softmax + rescale of acc_o ----
        Tensor acc_s_rc = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));

        // Coordinate view for masking (only built when needed).
        Tensor cS = local_tile(make_identity_tensor(make_shape(params.seqlen_q, params.seqlen_k)),
                               Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, n_block));
        Tensor tScS = thr_mma.partition_C(cS);
        Tensor tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));

        CUTE_UNROLL
        for (int r = 0; r < size<0>(acc_s_rc); ++r) {
            if (In_mask) {
                if (Is_causal) {
                    int col_limit = min(get<0>(tScS_rc(r, _0{})) + 1, params.seqlen_k);
                    CUTE_UNROLL
                    for (int c = 0; c < size<1>(acc_s_rc); ++c)
                        if (get<1>(tScS_rc(_0{}, c)) >= col_limit) acc_s_rc(r, c) = -INFINITY;
                } else {
                    CUTE_UNROLL
                    for (int c = 0; c < size<1>(acc_s_rc); ++c)
                        if (get<1>(tScS_rc(_0{}, c)) >= params.seqlen_k) acc_s_rc(r, c) = -INFINITY;
                }
            }

            float m_prev = row_max(r);
            float m_cur  = -INFINITY;
            CUTE_UNROLL
            for (int c = 0; c < size<1>(acc_s_rc); ++c) m_cur = fmaxf(m_cur, acc_s_rc(r, c));
            m_cur = quad_allreduce_max(m_cur);
            if (!Is_first) m_cur = fmaxf(m_prev, m_cur);
            if (Is_causal && m_cur == -INFINITY) m_cur = 0.f;  // fully-masked row

            const float m_scaled = m_cur * params.scale_log2;
            float s_cur = 0.f;
            CUTE_UNROLL
            for (int c = 0; c < size<1>(acc_s_rc); ++c) {
                float e = flash_exp2(acc_s_rc(r, c) * params.scale_log2 - m_scaled);
                acc_s_rc(r, c) = e;
                s_cur += e;
            }
            if (!Is_first) {
                float corr = flash_exp2(m_prev * params.scale_log2 - m_scaled);
                s_cur += row_sum(r) * corr;
                CUTE_UNROLL
                for (int c = 0; c < size<1>(acc_o_rc); ++c) acc_o_rc(r, c) *= corr;
            }
            row_max(r) = m_cur;
            row_sum(r) = s_cur;
        }

        // ---- P (bf16) and O += P * V ----
        Tensor rP = make_fragment_like<Element>(acc_s);
        CUTE_UNROLL
        for (int i = 0; i < size(acc_s); ++i) rP(i) = static_cast<Element>(acc_s(i));
        Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs(rP.layout())); // (MMA,MMA_M,MMA_K)

        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tOrVt_view(_, _, _0{}));
        CUTE_UNROLL
        for (int k = 0; k < size<2>(tOrP); ++k) {
            int k_next = (k + 1) % size<2>(tOrP);
            cute::copy(smem_tiled_copy_V, tOsVt(_, _, k_next), tOrVt_view(_, _, k_next));
            cute::gemm(tiled_mma, tOrP(_, _, k), tOrVt(_, _, k), acc_o);
        }
    };

    // ---- Main loop: high -> low n_block (matches causal masking order) ----
    // First `mask_steps` tiles carry masking; the rest are interior (no mask).
    constexpr int static_mask_steps = Is_causal ? (kBlockM + kBlockN - 1) / kBlockN : 1;

    int n_tile = 0;
    // Masked prefix.
    CUTE_NO_UNROLL
    for (; n_tile < static_mask_steps && n_tile < n_block_max; ++n_tile) {
        int n_block = n_block_max - n_tile - 1;
        if (n_tile == 0)
            compute_one_n_block(n_block, cute::true_type{},  cute::true_type{});
        else
            compute_one_n_block(n_block, cute::false_type{}, cute::true_type{});
    }
    // Interior (unmasked) tiles.
    CUTE_NO_UNROLL
    for (; n_tile < n_block_max; ++n_tile) {
        int n_block = n_block_max - n_tile - 1;
        compute_one_n_block(n_block, cute::false_type{}, cute::false_type{});
    }

    // -----------------------------------------------------------------------
    // Epilogue: normalize by row_sum, then round-trip O through smem for a
    // coalesced, predicated store.
    // -----------------------------------------------------------------------
    CUTE_UNROLL
    for (int r = 0; r < size<0>(acc_o_rc); ++r) {
        float s = quad_allreduce_sum(row_sum(r));
        float inv = (s == 0.f || s != s) ? 1.f : flash_rcp(s);
        CUTE_UNROLL
        for (int c = 0; c < size<1>(acc_o_rc); ++c) acc_o_rc(r, c) *= inv;
    }

    Tensor rO = make_fragment_like<Element>(acc_o);
    CUTE_UNROLL
    for (int i = 0; i < size(acc_o); ++i) rO(i) = static_cast<Element>(acc_o(i));

    // Reuse the Q smem region for O (Q is done).
    Tensor sO = make_tensor(make_smem_ptr(shared.smem_q.data()), typename Traits::SmemLayoutQ{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    __syncthreads();                       // ensure V no longer needed before clobbering
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    typename Traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_fragment_like(tOgO);
    __syncthreads();                       // all threads finished writing sO
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Predicated store to gmem.
    Tensor cO = local_tile(make_identity_tensor(make_shape(params.seqlen_q, params.d)),
                           Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tOgO); ++k) tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < params.d;
    CUTE_UNROLL
    for (int m = 0; m < size<1>(tOgO); ++m) {
        if (get<0>(tOcO(_0{}, m, _0{})) < params.seqlen_q) {
            CUTE_UNROLL
            for (int k = 0; k < size<2>(tOgO); ++k)
                if (tOpO(k)) cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), tOgO(_, m, k));
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher.
// ---------------------------------------------------------------------------
template <class Traits, bool Is_causal>
void run_flash_fwd(const Flash_params& params, cudaStream_t stream) {
    dim3 grid(cute::ceil_div(params.seqlen_q, Traits::kBlockM), params.b, params.h);
    dim3 block(Traits::kNThreads);
    int smem = Traits::kSmemSize;
    auto kern = &flash_fwd_kernel<Traits, Is_causal>;
    if (smem >= 48 * 1024) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    kern<<<grid, block, smem, stream>>>(params);
}

// ===========================================================================
// Test harness: fp32 CPU reference vs the kernel. Independent of PyTorch.
// ===========================================================================
#ifndef FLASH_NO_MAIN

template <class Element>
static void fill_small_ints(std::vector<Element>& h, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> d(-2, 1);   // matches the DSL's random_(-2, 2)
    for (auto& x : h) x = static_cast<Element>(float(d(rng)));
}

template <class Element>
static void cpu_reference(const std::vector<Element>& Q, const std::vector<Element>& K,
                          const std::vector<Element>& V, std::vector<float>& O,
                          int B, int H, int Sq, int Sk, int D, float scale, bool causal) {
    auto idx = [&](int b, int s, int h, int d, int S) { return ((int64_t(b) * S + s) * H + h) * D + d; };
    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
    for (int i = 0; i < Sq; ++i) {
        std::vector<float> logits(Sk, -INFINITY);
        float mx = -INFINITY;
        for (int j = 0; j < Sk; ++j) {
            if (causal && j > i) continue;
            float acc = 0.f;
            for (int d = 0; d < D; ++d)
                acc += float(Q[idx(b, i, h, d, Sq)]) * float(K[idx(b, j, h, d, Sk)]);
            logits[j] = acc * scale;
            mx = fmaxf(mx, logits[j]);
        }
        float denom = 0.f;
        for (int j = 0; j < Sk; ++j) if (logits[j] > -INFINITY) denom += expf(logits[j] - mx);
        for (int d = 0; d < D; ++d) {
            float acc = 0.f;
            for (int j = 0; j < Sk; ++j)
                if (logits[j] > -INFINITY) acc += expf(logits[j] - mx) * float(V[idx(b, j, h, d, Sk)]);
            O[idx(b, i, h, d, Sq)] = (denom > 0.f) ? acc / denom : 0.f;
        }
    }
}

template <class Element, int kHeadDim, int kBlockM, int kBlockN, int kNWarps>
static bool run_case(int B, int H, int Sq, int Sk, int D, float scale, bool causal) {
    using Traits = Fwd_traits<Element, kHeadDim, kBlockM, kBlockN, kNWarps>;
    const int64_t nQ = int64_t(B) * Sq * H * D;
    const int64_t nK = int64_t(B) * Sk * H * D;
    std::vector<Element> hQ(nQ), hK(nK), hV(nK), hO(nQ);
    fill_small_ints(hQ, 1); fill_small_ints(hK, 2); fill_small_ints(hV, 3);

    Element *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, nQ * sizeof(Element)); cudaMalloc(&dK, nK * sizeof(Element));
    cudaMalloc(&dV, nK * sizeof(Element)); cudaMalloc(&dO, nQ * sizeof(Element));
    cudaMemcpy(dQ, hQ.data(), nQ * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), nK * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), nK * sizeof(Element), cudaMemcpyHostToDevice);

    Flash_params p{};
    p.q_ptr = dQ; p.k_ptr = dK; p.v_ptr = dV; p.o_ptr = dO;
    p.b = B; p.h = H; p.d = D; p.seqlen_q = Sq; p.seqlen_k = Sk;
    p.q_row_stride = int64_t(H) * D; p.q_head_stride = D; p.q_batch_stride = int64_t(Sq) * H * D;
    p.k_row_stride = int64_t(H) * D; p.k_head_stride = D; p.k_batch_stride = int64_t(Sk) * H * D;
    p.v_row_stride = int64_t(H) * D; p.v_head_stride = D; p.v_batch_stride = int64_t(Sk) * H * D;
    p.o_row_stride = int64_t(H) * D; p.o_head_stride = D; p.o_batch_stride = int64_t(Sq) * H * D;
    p.scale_log2 = scale * 1.4426950408889634f;

    if (causal) run_flash_fwd<Traits, true >(p, 0);
    else        run_flash_fwd<Traits, false>(p, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("  CUDA error: %s\n", cudaGetErrorString(err)); return false; }

    cudaMemcpy(hO.data(), dO, nQ * sizeof(Element), cudaMemcpyDeviceToHost);
    std::vector<float> ref(nQ, 0.f);
    cpu_reference(hQ, hK, hV, ref, B, H, Sq, Sk, D, scale, causal);

    double max_abs = 0.0; int bad = 0;
    for (int64_t i = 0; i < nQ; ++i) {
        double diff = std::fabs(double(float(hO[i])) - double(ref[i]));
        max_abs = std::max(max_abs, diff);
        if (diff > 2e-2) ++bad;
    }
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    printf("  max|abs err| = %.4g, mismatches(>2e-2) = %d\n", max_abs, bad);
    return bad == 0;
}

int main() {
    using Element = cutlass::bfloat16_t;
    const float scale = 0.5f;
    bool ok = true;
    printf("[non-causal] B=2 H=4 Sq=Sk=256 D=128, m=n=128, 4 warps\n");
    ok &= run_case<Element, 128, 128, 128, 4>(2, 4, 256, 256, 128, scale, false);
    printf("[causal]     B=2 H=4 Sq=Sk=256 D=128, m=n=128, 4 warps\n");
    ok &= run_case<Element, 128, 128, 128, 4>(2, 4, 256, 256, 128, scale, true);
    printf("[non-causal] ragged Sq=200 Sk=200 D=128 (partial tiles)\n");
    ok &= run_case<Element, 128, 128, 128, 4>(1, 2, 200, 200, 128, scale, false);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
#endif  // FLASH_NO_MAIN

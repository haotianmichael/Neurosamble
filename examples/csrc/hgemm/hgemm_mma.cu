#include <cstdint>
#include <cuda_runtime.h>
#include "common/common.h"
#include "common/tester.h"
#include "common/util.h"
#include "ptx.cuh"
#include "swizzle.cuh"

using namespace nvcuda;
__device__ __forceinline__ void ld_st_128bit(void *dst, void *src) {
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
} 
__device__ __forceinline__ void ld_st_32bit(void *dst, void *src) {
    *reinterpret_cast<half2 *>(dst) = *reinterpret_cast<half2 *>(src);
}

/*
    @ C = A * B
        * tileB[K][N]
        * ldmatrix_trans_sync
    @ C = A * B^T
        * 一般可以先进入kernel之前就把B转置成B^T,kernel本身不变
    @ldmatrix+自定义uint32_t寄存器
    @preSM90: ld_st_128bit
    @SM90: stmatrix.sync
    @Requires  M % BM == 0, N % BN == 0, K % BK == 0.
*/
template <int MMA_M = 16, int MMA_N = 8, int MMA_K = 16,
          int MMA_TILE_M = 4, int MMA_TILE_N = 2,
          int WARP_TILE_M = 2, int WARP_TILE_N = 8,
          int K_STAGE = 4>
__global__ void hgemm_mma_m16n8k16_ldmatrix_kernel(
        const half *__restrict__ A, const half *__restrict__ B, 
        half *__restrict__ C, const int M, const int N, const int K) {

    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 128
    constexpr int BK = MMA_K; // 16
    const int K_NUM_TILES = (K + BK - 1) / BK;

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    // gmem->smem mapping: every thread moves one 16B (8-half) chunk of A and of B
    const int a_sm_m = tid / 2;
    const int a_sm_k = (tid % 2) * 8;
    const int b_sm_k = tid / 16;
    const int b_sm_n = (tid % 16) * 8;
    const int a_gm_m = by * BM + a_sm_m;
    const int b_gm_n = bx * BN + b_sm_n;

    // --- shared mem: A/B ring buffer; C aliases the SAME storage (epilogue reuse) ---
    extern __shared__ half smem[]; 
    half *s_a = smem;                       // [K_STAGE][BM][BK]
    half *s_b = smem + K_STAGE * BM * BK;   // [K_STAGE][BK][BN]
    half *s_c = smem;                       // [BM][BN], only after the K loop

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2] = {}; 

    // -------------------- prologue: prefetch first K_STAGE-1 tiles --------------------
    #pragma unroll
    for(int s = 0; s < K_STAGE - 1; s ++) {
        const int a_gm_k = s * BK + a_sm_k;
        const int b_gm_k = s * BK + b_sm_k;
        ptx::cp_async_cg<16>(&s_a[s * BM * BK + a_sm_m * BK + a_sm_k], &A[a_gm_m * K + a_gm_k]);
        ptx::cp_async_cg<16>(&s_b[s * BK * BN + b_sm_k * BN + b_sm_n], &B[b_gm_k * N + b_gm_n]);
        ptx::cp_async_commit_group();
    }

    // ------------------------------- main loop -------------------------------
    for(int k = 0; k < K_NUM_TILES; k ++) {
        // leave K_STAGE-2 groups in flight -> the tile we are about to consume is ready
        ptx::cp_async_wait_group<K_STAGE - 2>();
        __syncthreads();

        const int rs = k % K_STAGE;
        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

        #pragma unroll
        for(int i = 0; i < WARP_TILE_M; i ++) {
            int row = warp_m * WARP_TILE_M * MMA_M + i * MMA_M + lane_id % 16;
            int col = (lane_id / 16) * 8;
            ptx::ldmatrix_sync(RA[i], &s_a[rs * BM * BK + row * BK + col]);
        }
        #pragma unroll
        for(int j = 0; j < WARP_TILE_N; j ++) {
            int row = lane_id % 16;
            int col = warp_n * WARP_TILE_N * MMA_N + j * MMA_N;
            ptx::ldmatrix_trans_sync(RB[j], &s_b[rs * BK * BN + row * BN + col]);
        }
        #pragma unroll
        for(int i = 0; i < WARP_TILE_M; i ++)
            #pragma unroll
                for(int j = 0; j < WARP_TILE_N; j ++)
                    ptx::mma_sync_m16n8k16(RC[i][j], RA[i], RB[j]);

        // prefetch tile (k + K_STAGE - 1) into buffer (k-1)%K_STAGE
        int nk = k + K_STAGE - 1;
        if(nk < K_NUM_TILES) {
            int ws = nk % K_STAGE;
            int a_gm_k = nk * BK + a_sm_k;
            int b_gm_k = nk * BK + b_sm_k;
            ptx::cp_async_cg<16>(&s_a[ws * BM * BK + a_sm_m * BK + a_sm_k], &A[a_gm_m * K + a_gm_k]);
            ptx::cp_async_cg<16>(&s_b[ws * BK * BN + b_sm_k * BN + b_sm_n], &B[b_gm_k * N + b_gm_n]);
        }
        // commit even when nk is out of range -> empty group keeps wait_group accounting uniform
        ptx::cp_async_commit_group();
    }
   
    // ------------------------------- epilogue -------------------------------
    // all A/B ldmatrix reads are done -> safe to clobber the union with C
    __syncthreads();
  
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            int row = warp_m * WARP_TILE_M * MMA_M + i * MMA_M + lane_id / 4;
            int col = warp_n * WARP_TILE_N * MMA_N + j * MMA_N + (lane_id % 4) * 2;
            *reinterpret_cast<uint32_t *>(&s_c[row * BN + col])       = RC[i][j][0];
            *reinterpret_cast<uint32_t *>(&s_c[(row + 8) * BN + col]) = RC[i][j][1];
        }
    __syncthreads();

    // s_c[128][128] -> C, 128-bit vectorized & coalesced, independent of warp layout
    for (int idx = tid; idx < (BM * BN) / 8; idx += (blockDim.x*blockDim.y)) {
        int row = (idx * 8) / BN;
        int col = (idx * 8) % BN;
        int gm  = by * BM + row;
        int gn  = bx * BN + col;
        *reinterpret_cast<uint4 *>(&C[gm * N + gn]) =
            *reinterpret_cast<uint4 *>(&s_c[row * BN + col]);
    }
    return;
}
void hgemm_mma_m16n8k16_ldmatrix(half *A, half *B, half *C, const int M, const int N, const int K) {

    constexpr int MMA_M = 16, MMA_K = 16, MMA_N = 8;
    constexpr int MMA_TILE_M = 4, MMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2, WARP_TILE_N = 8;
    constexpr int K_STAGE = 4;

    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 128
    constexpr int BK = MMA_K;                            // 16

    constexpr int ab_bytes = K_STAGE * (BM * BK + BK * BN) * (int)sizeof(half);
    constexpr int c_bytes  = BM * BN * (int)sizeof(half);
    constexpr int smem_bytes = ab_bytes > c_bytes ? ab_bytes : c_bytes; // 32 KB here

    auto kernel = hgemm_mma_m16n8k16_ldmatrix_kernel<
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, K_STAGE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    dim3 block(16, 16);
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
    kernel<<<grid, block, smem_bytes>>>(A, B, C, M, N, K);
    
    return;
}

/*
    @ C = A * B^T
        * wmma::col_major
        * ldmatrix_sync
        * swap R1 and R2
    @bank conflict solved by swizzle
    @16x16 half
    @wmma寄存器
*/
__global__ void hgemm_mma16x16_swizzle_kernel(half *A, half *B, half *C) {

    __shared__ half tileA[16*16];
    __shared__ half tileB[16*16];
    __shared__ half tileC[16*16];

    int tx = threadIdx.x;
    uint32_t gAddr = tx * 8;
    auto g2sAddr = swizzle<3, 1, 3>(gAddr);
    ld_st_128bit(tileA + g2sAddr, A+gAddr);
    ld_st_128bit(tileB + g2sAddr, B+gAddr);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    uint32_t rAddr = (tx % 16) * 16 + (tx / 16) * 8;
    auto r2sAddr = swizzle<3, 1, 3>(rAddr);

    ptx::ldmatrix_sync(a_frag.x, tileA + r2sAddr);
    ptx::ldmatrix_sync(b_frag.x, tileB +r2sAddr);

    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;

    // mma_sync(c_frag, a_frag, b_frag, c_frag);
    ptx::mma_sync_m16n8k16(c_frag.x, a_frag.x, b_frag.x);
    ptx::mma_sync_m16n8k16(c_frag.x + 4, a_frag.x, b_frag.x + 4);
    ptx::stmatrix_sync(tileC + r2sAddr, c_frag.x);

    ld_st_128bit(C + gAddr, tileC + g2sAddr);
}
void hgemm_mma16x16_swizzle(half *A, half *B, half*C, int M, int N, int K) {
    dim3 block(32);
    dim3 grid(1);
    hgemm_mma16x16_swizzle_kernel<<<grid, block>>>(A, B, C);
    return;
}

/*
    @ C = A * B^T
        * wmma::col_major
        * ldmatrix_sync
        * swap R1 and R2
    @bank conflict solved by swizzle
    @16x64 half
    @wmma寄存器
 */
__global__ void hgemm_mma16x64_swizzle_kernel(half *A, half *B, half *C) {
    __shared__ half smem_a[16 * 64];
    __shared__ half smem_b[16 * 64];
    __shared__ half smem_c[16 * 64];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * blockDim.x;
    // swizzle load A and B
    constexpr int stride = 64;
    int gRow = tid * 8 / stride;
    int gCol = tid * 8 % stride;

    int g2sRow = gRow;
    // [xxxx] [xxx] [xxx]
    // [16row] [8col] [8fp16]
    int g2sCol = gCol ^ ((gRow & 0x7) << 3);

    ld_st_128bit(smem_a + g2sRow * stride + g2sCol, A + tid * 8);
    ld_st_128bit(smem_b + g2sRow * stride + g2sCol, B + tid * 8);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    // swizzle load frag a and b
    int rRow = tx % 16;
    int rCol = (ty * 2 + tx / 16) * 8;
    int r2sRow = rRow;
    int r2sCol = rCol ^ ((rRow & 0x7) << 3);
    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sRow * stride + r2sCol);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sRow * stride + r2sCol);
    // swap R1 and R2 of B, this is required by B's layout, more info see PTX
    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;
    // calc and store
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    // store_matrix_sync(smem_c + 16 * ty, c_frag, 16 * 4, mem_row_major);
    ptx::stmatrix_sync(smem_c + 16 * ty + (tx % 16) * 64 + (tx / 16) * 8,
                       c_frag.x);
    __syncthreads();
    ld_st_128bit(C + 8 * tid, smem_c + 8 * tid);
}

/**
 * \brief 2 patterns are resolved in this kernel, one is a row 1x256 regarded as
 * 16x16, the other is block 16x16
 *
 * \note this kernel has serious LDSM bank
 * conflicts calculated as follows
 *
 * Pattern 1: 1x256 regarded as 16x16, each 1x256 has 4 bank conflicts, result
 * in 4x16(rows)x2(matrix A/B) = 128 bank conflicts
 *
 * Pattern 2: 16x16 block, each 16x16 has 7x4 bank conflicts, result in
 * 7x4x16(blocks)x2(matrix A/B) = 896 bank conflicts
 *
 * Total bank conflicts = 128 + 896 = 1024
 */
__global__ void mma_multi_pattern_simple(half *A, half *B, half *C) {
    __shared__ half smem_a[16 * 256];
    __shared__ half smem_b[16 * 256];
    __shared__ half smem_c[16 * 256];

    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y; // 0-15

    int tid = tx + ty * blockDim.x;
    // TODO: swizzle load A and B
    ld_st_128bit(smem_a + tid * 8, A + tid * 8);
    ld_st_128bit(smem_b + tid * 8, B + tid * 8);

    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // 1x256 regarded as 16x16, compute C = A * B^T
    load_matrix_sync(a_frag, smem_a + 256 * ty, 16);
    load_matrix_sync(b_frag, smem_b + 256 * ty, 16);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(smem_c + 256 * ty, c_frag, 16, wmma::mem_row_major);

    __syncthreads();

    // 16x16 block, compute C = A * B^T
    load_matrix_sync(a_frag, smem_a + 16 * ty, 256);
    load_matrix_sync(b_frag, smem_b + 16 * ty, 256);

    fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(smem_c + 16 * ty, c_frag, 256, wmma::mem_row_major);

    __syncthreads();

    ld_st_128bit(C + tid * 8, smem_c + tid * 8);
}

__global__ void mma_multi_pattern_swizzle(half *A, half *B, half *C) {
    __shared__ half smem_a[16 * 256];
    __shared__ half smem_b[16 * 256];
    __shared__ half smem_c[16 * 256];

    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y; // 0-15
    int tid = tx + ty * blockDim.x;

    // swizzle load A and B
    // [xxxx]    [xxxxx]    [xxx]
    // [16rows]  [32cols]    [8fp16]
    // split cols into 4 groups:
    // [xxxx]    [xx] [xxx]     [xxx]

    uint32_t gAddr = tid * 8;
    auto g2sAddr = swizzle<3, 2, 3>(swizzle<5, 3, 3>(gAddr));
    ld_st_128bit(smem_a + g2sAddr, A + gAddr);
    ld_st_128bit(smem_b + g2sAddr, B + gAddr);

    __syncthreads();

    using namespace nvcuda::wmma;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    // 1x256 regarded as 16x16
    uint32_t rAddr = ty * 256 + (tx % 16) * 16 + (tx / 16 * 8);
    auto r2sAddr = swizzle<3, 2, 3>(swizzle<5, 3, 3>(rAddr));

    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sAddr);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sAddr);

    half2 tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;

    wmma::fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(smem_c + 256 * ty, c_frag, 16, mem_row_major);
    __syncthreads();
    // 16x16 blocks
    rAddr = ty * 16 + (tx % 16) * 256 + (tx / 16 * 8);
    r2sAddr = swizzle<3, 2, 3>(swizzle<5, 3, 3>(rAddr));

    ptx::ldmatrix_sync(a_frag.x, smem_a + r2sAddr);
    ptx::ldmatrix_sync(b_frag.x, smem_b + r2sAddr);

    tmp = HALF2(b_frag.x[2]);
    HALF2(b_frag.x[2]) = HALF2(b_frag.x[4]);
    HALF2(b_frag.x[4]) = tmp;

    wmma::fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(smem_c + 16 * ty, c_frag, 256, mem_row_major);
    __syncthreads();
    ld_st_128bit(C + tid * 8, smem_c + tid * 8);
}

int main() {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    const int opt = 1;
    if(opt == 1) {
        tester.evaluate(hgemm_mma_m16n8k16_ldmatrix, "hgemm_mma_m16n8k16_ldmatrix_kernel");
    }
    
    return 0;
}
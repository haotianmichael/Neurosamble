#include <cstdint>
#include <cuda_runtime.h>
#include "common/common.h"
#include "common/tester.h"
#include "common/util.h"
#include "ptx.cuh"

using namespace nvcuda;
/*
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;
void load_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm, layout_t layout);
void store_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &a, fragment<...> &b, const fragment<...> &c, bool staf=false);
*/

/* 
   @ C = A * B
   @naive wmma version
*/
template<unsigned int WMMA_M = 16,
         unsigned int WMMA_N = 16,
         unsigned int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_kernel_naive(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {

    unsigned int row = blockIdx.y * WMMA_M;
    unsigned int col = blockIdx.x * WMMA_N;
    if(row >= M || col >= N) return;

    half *A_bck = A + row * K;
    half *B_bck = B + col;
    half *C_bck = C + row * N + col;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;

#pragma unroll
    for(int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(A_frag, A_bck + k, K);
        wmma::load_matrix_sync(B_frag, B_bck + k * N, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    if(row + WMMA_M <= M && col + WMMA_N <= N) {
        wmma::store_matrix_sync(C_bck, C_frag, N, wmma::mem_row_major);
    }
}
void hgemm_wmma_m16n16k16_naive(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_N = 16;
    dim3 block(32);
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    hgemm_wmma_m16n16k16_kernel_naive<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(A, B, C, M, N, K); 
}

/*
opt:
    @ C = A * B
        * tileB[K][N]
        * wmma::col_major
            * wmma::load_matrix_sync包含trans B的布局
    1. block: 8个warp处理一块Block数据
    2. mma4x2: 数据(128x128->4x2个[32,64]的tile)
    3. warp2x4: wmma操作(一个tile用2x4个fragment处理->每个warp处理8个fragment)
    4. async data.mov
    5. double buffer
*/
template<unsigned int WMMA_M = 16,
         unsigned int WMMA_N = 16,
         unsigned int WMMA_K = 16,
         unsigned int WMMA_TILE_M = 4,
         unsigned int WMMA_TILE_N = 2,
         unsigned int WARP_TILE_M = 2,
         unsigned int WARP_TILE_N = 4>
__global__ void hgemm_wmma_m16n16k16_kernel_opt(half *A, half *B, half *C, const int M, const int N, const int K) {

    const int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 128
    const int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 128
    const int BK = WMMA_K;
    const int K_NUM_TILES = (K + BK - 1) / BK;

    unsigned int ty = threadIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int tid = ty * blockDim.x + tx;
    unsigned int warp_id = tid / 32;
    unsigned int warp_m = warp_id / 2;
    unsigned int warp_n = warp_id % 2;

    const int load_smem_a_m = tid / 2;
    const int load_smem_a_k = (tid % 2) * 8;
    const int load_smem_b_k = tid / 16;
    const int load_smem_b_n = (tid % 16) * 8;

    const int load_gmem_a_m = blockIdx.y * BM + load_smem_a_m;
    const int load_gmem_b_n = blockIdx.x * BN + load_smem_b_n;

    __shared__ half tileA[2][BM][BK], tileB[2][BK][BN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_TILE_M][WARP_TILE_N];
    for(int i = 0; i < WARP_TILE_M; i ++) {
        for(int j = 0; j < WARP_TILE_N; j ++) {
            wmma::fill_fragment(C_frag[i][j], 0.0); 
        }
    }
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag[WARP_TILE_N];
    unsigned int write_stage = 0;
    {
        /*
            Glm Access[load]:
                8half/thr -> 16Bytes/thr -> cp.async.cg.shared.global 
        */
        const int load_gmem_a_k = load_smem_a_k;
        const int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        const int load_gmem_b_k = load_smem_b_k;
        const int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        ptx::cp_async_cg<16>(&tileA[write_stage][load_smem_a_m][load_smem_a_k], &A[load_gmem_a_addr]);
        ptx::cp_async_cg<16>(&tileB[write_stage][load_smem_b_k][load_smem_b_n], &B[load_gmem_b_addr]); 
        ptx::cp_async_commit_group();
        ptx::cp_async_wait_group<0>();
        write_stage ^= 1;
    }
    __syncthreads();

    for(int s = 1; s < K_NUM_TILES; s ++) {
        const int load_gmem_a_k = s * BK + load_smem_a_k;
        const int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        const int load_gmem_b_k = s * BK + load_smem_b_k;
        const int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        ptx::cp_async_cg<16>(&tileA[write_stage][load_smem_a_m][load_smem_a_k], &A[load_gmem_a_addr]);
        ptx::cp_async_cg<16>(&tileB[write_stage][load_smem_b_k][load_smem_b_n], &B[load_gmem_b_addr]); 
        write_stage ^= 1;

        /*
            Shm Access[load]: 
                wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 
        */
        for(int i = 0; i < WARP_TILE_M; i ++) {
            wmma::load_matrix_sync(A_frag[i], &tileA[write_stage][warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M][0], BK);
        }
        for(int j = 0; j < WARP_TILE_N; j ++) {
            wmma::load_matrix_sync(B_frag[j], &tileB[write_stage][0][warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N], BN);
        }
        /*
            Compute:
                 wmma.mma.sync.aligned.row.m16n16k16.shared.f16
        */
        for(int i = 0; i < WARP_TILE_M; i ++) {
            for(int j = 0; j < WARP_TILE_N; j ++) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
        ptx::cp_async_commit_group();
        ptx::cp_async_wait_group<0>();
        __syncthreads();
    }
    {
        write_stage ^= 1;
        for(int i = 0; i < WARP_TILE_M; i ++) {
            wmma::load_matrix_sync(A_frag[i], &tileA[write_stage][warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M][0], BK);
        }
        for(int j = 0; j < WARP_TILE_N; j ++) {
            wmma::load_matrix_sync(B_frag[j], &tileB[write_stage][0][warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N], BN);
        }
        for(int i = 0; i < WARP_TILE_M; i ++) {
            for(int j = 0; j < WARP_TILE_N; j ++) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
    }
    /*
        Glm Access[Store]:
            wmma.store.d.sync.aligned.row.m16n16k16.global.f16
    */
    for(int i = 0; i < WARP_TILE_M; i ++){
        for(int j = 0; j < WARP_TILE_N; j ++) {
            const int store_matrix_gmem_m = blockIdx.y * BM + warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M;
            const int store_matrix_gmem_n = blockIdx.x * BN + warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N;
            wmma::store_matrix_sync(C + store_matrix_gmem_m * N + store_matrix_gmem_n, C_frag[i][j], N, wmma::mem_row_major); 
        }
    }

    return;
}
void hgemm_wmma_m16n16k16_opt(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned K) {

    constexpr int WMMA_M = 16; 
    constexpr int WMMA_K = 16; 
    constexpr int WMMA_N = 16; 
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;

    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N*WMMA_TILE_N*WARP_TILE_N), div_ceil(M, WMMA_TILE_M*WMMA_M*WARP_TILE_M));
    hgemm_wmma_m16n16k16_kernel_opt<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);

    return;
}

int main() {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    const int opt = 1;
    if(opt == 0) {
        tester.evaluate(hgemm_wmma_m16n16k16_naive, "hgemm_wmma_m16n16k16_kernel_naive");
    }else if(opt == 1) {
        tester.evaluate(hgemm_wmma_m16n16k16_opt, "hgemm_wmma_m16n16k16_kernel_opt");
    }
    return 0;
}
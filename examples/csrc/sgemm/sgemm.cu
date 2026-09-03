#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <pthread.h>
#include <random>
#include <vector>


void random_matrix(int m, int n, std::vector<float> &A) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 2.0f);
    for(int i = 0; i < m; i ++)
        for(int j = 0; j < n; j ++) {
            A[i * n + j] = dist(rng);
        }
}
void cpu_sgemm(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, const int m, const int k, const int n) {

    for(int i = 0; i < m; i ++) {
        for(int j = 0; j < n; j ++) {
            float res = 0.0f;
            for(int _k = 0; _k < k; _k++) {
                res += A[i * k + _k] * B[_k * n + j]; 
            }
            C[i * n + j] = res;
        }
    }
}
float compare_matrices(int m, int n, std::vector<float> &gpu_C, std::vector<float> &cpu_C) {
    float max_diff = 0, diff;
    int printed = 0;

    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j ++) {
            diff = std::abs(gpu_C[i * n + j] - cpu_C[i * n + j]);
            max_diff = (diff > max_diff) ? diff : max_diff;
            if(printed == 0) {
                if(max_diff > 0.5f || max_diff < -0.5f) {
                    printf("\n error: i %d j %d\ncpu: %f gpu: %f\n", i, j, cpu_C[i*n+j], gpu_C[i*n+j]);
                    return max_diff;
                }
            }
        }
    std::cout << "right" << std::endl;
    return 0;
}
/*
navie global_mem: 
*/
__global__ void cuda_sgemm_naive(float *A, float *B, float *C, const int M, const int N, const int K) {

    float *A_bck = A + blockIdx.x * K * blockDim.x;
    float *B_bck = B + blockIdx.y * blockDim.y;

    float res = 0.0f;
    for(int k = 0; k < K; k++) {
        res += A_bck[threadIdx.x * K + k] * B_bck[threadIdx.y + k * N];
    }
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    C[x * N + y] = res;
    return;
}

/* 
opt:
 @ C = A * B
 1. global mem
 2. tiled shared mem
 3. multiTask per thread
 4. 向量化加载
 5. 转置
 6. 外积
 7. double buffer
*/
__device__ __forceinline__ void ld_st_128bits(void *dst, void *src) {
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
}
template<unsigned int BLOCK_SIZE_M = 128,
         unsigned int BLOCK_SIZE_K = 8,
         unsigned int BLOCK_SIZE_N = 128,
         unsigned int THREAD_SIZE_X = 8,
         unsigned int THREAD_SIZE_Y = 8> 
__global__ void cuda_sgemm_opt(float *A, float *B, float *C, const int M, const int N, const int K) {

    const int BM = BLOCK_SIZE_M;
    const int BN = BLOCK_SIZE_N;
    const int BK = BLOCK_SIZE_K;

    const int K_NUM_TILES = (K + BK - 1) / BK;
    unsigned int ty = threadIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int tid = ty * blockDim.x + tx;

    const int load_smem_a_m = tid / 2;
    const int load_smem_a_k = (tid % 2) * 4;
    const int load_smem_b_k = tid / 32;
    const int load_smem_b_n = (tid % 32) * 4;

    const int load_gmem_a_m = blockIdx.y * BM + load_smem_a_m;
    const int load_gmem_b_n = blockIdx.x * BN + load_smem_b_n;

    if(load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    __shared__ float tileA[2][BK][BM];
    __shared__ float tileB[2][BK][BN];
    float regA[THREAD_SIZE_Y] = {0.0f};
    float regB[THREAD_SIZE_X] = {0.0f};
    float res[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};
    float A_load[4] = {0.0f};

    unsigned int write_stage = 0;
    {
        /*
         Glm Access[load]:
            4float/thr -> 16Bytes/thr  ->  ld.global.v4.f32/thr -> st.shared.v4.f32/thr
        */
        const int load_gmem_a_k = load_smem_a_k;
        const int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        ld_st_128bits(&A_load[0], &A[load_gmem_a_addr]);
        tileA[write_stage][load_smem_a_k][load_smem_a_m] = A_load[0];
        tileA[write_stage][load_smem_a_k + 1][load_smem_a_m] = A_load[1];
        tileA[write_stage][load_smem_a_k + 2][load_smem_a_m] = A_load[2];
        tileA[write_stage][load_smem_a_k + 3][load_smem_a_m] = A_load[3];
        const int load_gmem_b_k = load_smem_b_k;
        const int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        ld_st_128bits(&tileB[write_stage][load_smem_b_k][load_smem_b_n], &B[load_gmem_b_addr]);
    }
    write_stage ^= 1;
    __syncthreads();

    for(int s = 1; s < K_NUM_TILES; s ++) {
        const int load_gmem_a_k = s * BK + load_smem_a_k;
        const int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        ld_st_128bits(&A_load[0], &A[load_gmem_a_addr]);
        tileA[write_stage][load_smem_a_k][load_smem_a_m] = A_load[0];
        tileA[write_stage][load_smem_a_k + 1][load_smem_a_m] = A_load[1];
        tileA[write_stage][load_smem_a_k + 2][load_smem_a_m] = A_load[2];
        tileA[write_stage][load_smem_a_k + 3][load_smem_a_m] = A_load[3];
        const int load_gmem_b_k = s * BK + load_smem_b_k;
        const int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        ld_st_128bits(&tileB[write_stage][load_smem_b_k][load_smem_b_n], &B[load_gmem_b_addr]);

        write_stage ^= 1;
        for(int k = 0; k < BK; k ++) {
            /*
                Shm Access[load]:
                    ld.shared.v4.f32/thr 
            */
            ld_st_128bits(&regA[0], &tileA[write_stage][k][ty * THREAD_SIZE_Y]);
            ld_st_128bits(&regA[4], &tileA[write_stage][k][ty * THREAD_SIZE_Y + 4]);
            ld_st_128bits(&regB[0], &tileB[write_stage][k][tx * THREAD_SIZE_X]);
            ld_st_128bits(&regB[4], &tileB[write_stage][k][tx * THREAD_SIZE_X + 4]);
            /*
                Compute:
                    fma
            */
            for(int i = 0; i < THREAD_SIZE_Y; i++) {
                for(int j = 0; j < THREAD_SIZE_X; j ++) {
                    res[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    write_stage ^= 1;
    for(int k = 0; k < BK; k ++) {
        ld_st_128bits(&regA[0], &tileA[write_stage][k][ty * THREAD_SIZE_Y]);
        ld_st_128bits(&regA[4], &tileA[write_stage][k][ty * THREAD_SIZE_Y + 4]);
        ld_st_128bits(&regB[0], &tileB[write_stage][k][tx * THREAD_SIZE_X]);
        ld_st_128bits(&regB[4], &tileB[write_stage][k][tx * THREAD_SIZE_X + 4]);
        for(int i = 0; i < THREAD_SIZE_Y; i++) {
            for(int j = 0; j < THREAD_SIZE_X; j ++) {
                res[i][j] += regA[i] * regB[j];
            }
        }
    }
    /*
        Glm Access[Store]:
            st.global.v4.f32
    */
    for(int i = 0; i < THREAD_SIZE_Y; i ++) {
        const int store_matrix_gmem_m = blockIdx.y * BM + ty * THREAD_SIZE_Y + i;
        const int store_matrix_gmem_n = blockIdx.x * BN + tx * THREAD_SIZE_X;
        ld_st_128bits(&C[store_matrix_gmem_m * N + store_matrix_gmem_n], &res[i][0]);
        ld_st_128bits(&C[store_matrix_gmem_m * N + store_matrix_gmem_n + 4], &res[i][4]);
    }

    return;
}

int main() {
   
    unsigned int m = 512;
    unsigned int n = 512;
    constexpr int k = 256;
    
    std::vector<float> h_A(m * k), h_B(k * n), h_C(m * n), gpu_C(m * n);

    random_matrix(m, k, h_A);
    random_matrix(k, n, h_B);
    std::fill(h_C.begin(), h_C.end(), 0.0f);
    std::fill(gpu_C.begin(), gpu_C.end(), 0.0f);

    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, m * k * sizeof(float));
    cudaMalloc(&device_B, k * n * sizeof(float));
    cudaMalloc(&device_C, m * n * sizeof(float));

    cudaMemcpy(device_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    

    cpu_sgemm(h_A, h_B, h_C, m, k, n);
    const int opt = 1;
    constexpr int BLOCK = 16;
    if(opt == 0) {
        std::cout << "cuda_sgemm_naive" << std::endl;
        dim3 block(BLOCK, BLOCK);
        dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

        cuda_sgemm_naive<<<grid,block>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 
    }else if(opt == 1) {
        std::cout << "cuda_sgemm_opt" << std::endl;
        constexpr unsigned int BLOCK_SIZE_M = 128;
        constexpr unsigned int BLOCK_SIZE_N = 128;
        constexpr unsigned int BLOCK_SIZE_K = 8;
        constexpr unsigned int THREAD_SIZE_X = 8;
        constexpr unsigned int THREAD_SIZE_Y = 8;

        dim3 block_v8(16, 16);
        dim3 grid_v8((m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (n + BLOCK_SIZE_N- 1) / BLOCK_SIZE_N);
        cuda_sgemm_opt<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y><<<grid_v8, block_v8>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C);
    }

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    return 0;
}
#include <cstdint>
#include <mma.h>
#include <stdint.h>

#define  REG(val) (*reinterpret_cast<uint32_t *>(&(val)))
#define  HALF2(val) (*reinterpret_cast<half2 *>(&(val)))

// ---------------------------------------------------------------------------
// PTX helpers (uint32_t overloads, SM80/89 — no stmatrix, no Hopper)
// ---------------------------------------------------------------------------
namespace ptx {
using fp16 = half;
/* FP16_T*/
__device__ __forceinline__ void ldmatrix_sync(fp16 *dst, void *addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(REG(dst[0])),
          "=r"(REG(dst[2])),
          "=r"(REG(dst[4])),
          "=r"(REG(dst[6]))
        : "l"(__cvta_generic_to_shared(addr)));
}

__device__ __forceinline__ void ldmatrix_trans_sync(fp16 *dst, void *addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(REG(dst[0])),
          "=r"(REG(dst[2])),
          "=r"(REG(dst[4])),
          "=r"(REG(dst[6]))
        : "l"(__cvta_generic_to_shared(addr)));
}

// C = A * B^T
__device__ __forceinline__ void mma_sync_m16n8k16(fp16 *c, fp16 *a, fp16 *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3, %4, %5}, "
                 "{%6, %7}, "
                 "{%8, %9};"
                 : "=r"(REG(c[0])), "=r"(REG(c[2]))
                 : "r"(REG(a[0])),
                   "r"(REG(a[2])),
                   "r"(REG(a[4])),
                   "r"(REG(a[6])),
                   "r"(REG(b[0])),
                   "r"(REG(b[2])),
                   "r"(0),
                   "r"(0));
}

/* UINT32_t*/
__device__ __forceinline__ void ldmatrix_sync(uint32_t *dst, void *addr) {
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" 
      : "=r"(REG(dst[0])), 
        "=r"(REG(dst[1])), 
        "=r"(REG(dst[2])), 
        "=r"(REG(dst[3])) 
      : "l"(__cvta_generic_to_shared(addr)));
}

__device__ __forceinline__ void ldmatrix_trans_sync(uint32_t *dst, void *addr) {
    asm volatile(
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];"
     : "=r"(REG(dst[0])),
       "=r"(REG(dst[1])) 
     : "l"(__cvta_generic_to_shared(addr)));
}

// D(f16) = A * B + C(f16),  accumulates into c
__device__ __forceinline__ void mma_sync_m16n8k16(uint32_t *c, uint32_t *a, uint32_t *b) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" 
      : "=r"(REG(c[0])), "=r"(REG(c[1])) 
      : "r"(REG(a[0])), "r"(REG(a[1])), "r"(REG(a[2])), "r"(REG(a[3])), "r"(REG(b[0])), "r"(REG(b[1])), "r"(REG(c[0])), "r"(REG(c[1])));
}

__device__ __forceinline__ void stmatrix_sync(half *dst, half *src) {
    // ! Ampere doesn't have stmatrix.sync, we should simulate it
    uint64_t private_addr = (uint64_t)dst;
    uint64_t shared_addr[4];
#pragma  unroll
    for(int i = 0; i < 4; i ++) {
        shared_addr[i] = __shfl_sync(0xFFFFFFFF, private_addr, i * 8 + threadIdx.x / 4);
    }
#pragma  unroll
    for(int i = 0; i < 4; i ++) {
        *(reinterpret_cast<half2 *>(shared_addr[i]) + threadIdx.x % 4) = 
            HALF2(src[2 * i]);
    }
}

// ca(cache all, L1+L2): support 4, 8, 16Bytes, cg(cache global, L2): only support 16Bytes.
template<int BYTES>
__device__ __forceinline__ void cp_async_ca(half *dst, const half *src) {
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: 
        "l"(__cvta_generic_to_shared(dst)),
        "l"(src),
        "n"(BYTES)
    );
}

template<int BYTES>
__device__ __forceinline__ void cp_async_cg(half *dst, const half *src) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" :: 
        "l"(__cvta_generic_to_shared(dst)),
        "l"(src),
        "n"(BYTES)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

}
/*
 * gpu_miner_cuda.cu — hash256 keccak256 CUDA miner
 *
 * Drop-in replacement for gpu_miner.c (OpenCL), same JSON stdout protocol.
 * gpu_worker.js will auto-prefer this binary when present.
 *
 * KEY OPTIMIZATIONS vs the OpenCL version:
 *   1. __constant__ memory for RC[24], challenge, prefix, difficulty
 *      -> L1 constant-cache broadcast; all threads in a warp read at reg speed
 *   2. #pragma unroll on the 24-round keccak-f loop
 *      -> eliminates branch + loop-counter overhead entirely
 *   3. Lanes 0-6 preloaded into registers before the work loop
 *      -> constant data loaded once; only counter (lane 7) changes per nonce
 *   4. Thread coarsening (WORK_PER_THREAD nonces per thread per launch)
 *      -> amortizes kernel-launch overhead; hides memory latency better
 *   5. __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS) guides register allocator
 *      -> maximises occupancy without spilling state to local memory
 *   6. Entire 25-lane keccak state lives in registers (no shared/global)
 *   7. atomicCAS for race-free winner selection across threads/blocks
 *   8. Word-wide (4xuint64) difficulty comparison instead of 32 byte-loops
 *
 * Build (Linux):
 *   bash build-gpu-cuda.sh              # auto-detects SM via nvidia-smi
 *   nvcc -O3 -arch=sm_89 -o src/gpu_miner_cuda src/gpu_miner_cuda.cu  (RTX 40xx)
 *   nvcc -O3 -arch=sm_86 -o src/gpu_miner_cuda src/gpu_miner_cuda.cu  (RTX 30xx)
 *   nvcc -O3 -arch=sm_75 -o src/gpu_miner_cuda src/gpu_miner_cuda.cu  (RTX 20xx)
 *
 * Stdout JSON messages:
 *   {"type":"device",   "platform":"CUDA", "device":"...", "cu":N, "batch_size":N}
 *   {"type":"progress", "hashes":"N", "nonce":"N"}
 *   {"type":"found",    "counter":"N", "hashes":"N"}
 *   {"type":"error",    "message":"..."}
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <cuda_runtime.h>

/* -- Tuning ---------------------------------------------------------------- */
// T4 prefers a slightly smaller block size for complex kernels to maximize occupancy and minimize register spilling.
// The optimal configuration is auto-tuned at startup!
#define MIN_BLOCKS_PER_SM   2   /* hint to __launch_bounds__ for reg alloc */

// Toggle for experimental Warp-Cooperative Keccak (Step 9)
// 1 = 1 Warp per Hash (32 threads share 1 state)
// 0 = 1 Thread per Hash (Default)
#define WARP_COOP 0

/* -- Constant memory (L1-cached, broadcast to all threads in warp) --------- */
__constant__ uint8_t  d_challenge[32];
__constant__ uint8_t  d_prefix[24];
__constant__ ulonglong4 d_difficulty_be_vec; /* difficulty as 4 big-endian u64 packed in vector */

#if WARP_COOP
__constant__ int d_rot[25] = {
    0,  44, 43, 21, 14,
    28, 20, 3,  45, 61,
    1,  6,  25, 8,  18,
    27, 36, 10, 15, 56,
    62, 55, 39, 41, 2
};
#endif

__constant__ uint64_t d_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

/* -- Helpers --------------------------------------------------------------- */

/* Inline PTX 64-bit rotate left.
 * T4 (Turing) supports shf.l.wrap.b32 (funnel shift).
 * A 64-bit rotate left by n (0 < n < 64) is achieved using two 32-bit funnel shifts.
 */
__device__ __forceinline__ uint64_t rotl64_ptx(uint64_t x, int n) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t res_lo, res_hi;
    if (n < 32) {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(res_lo) : "r"(lo), "r"(hi), "r"(n));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(res_hi) : "r"(hi), "r"(lo), "r"(n));
    } else if (n > 32) {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(res_lo) : "r"(hi), "r"(lo), "r"(n - 32));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(res_hi) : "r"(lo), "r"(hi), "r"(n - 32));
    } else { // n == 32
        res_lo = hi;
        res_hi = lo;
    }
    return ((uint64_t)res_hi << 32) | res_lo;
}

#define ROTL64(x, n) rotl64_ptx((x), (n))

/* Inline PTX LOP3 instruction for Keccak Chi step.
 * Computes: a ^ (~b & c) in a single instruction per 32-bit half.
 * The LUT truth table for this operation is 0x9A.
 */
__device__ __forceinline__ uint64_t chi64_ptx(uint64_t a, uint64_t b, uint64_t c) {
    uint32_t a_lo = (uint32_t)a, a_hi = (uint32_t)(a >> 32);
    uint32_t b_lo = (uint32_t)b, b_hi = (uint32_t)(b >> 32);
    uint32_t c_lo = (uint32_t)c, c_hi = (uint32_t)(c >> 32);
    uint32_t res_lo, res_hi;
    asm("lop3.b32 %0, %1, %2, %3, 0x9A;" : "=r"(res_lo) : "r"(a_lo), "r"(b_lo), "r"(c_lo));
    asm("lop3.b32 %0, %1, %2, %3, 0x9A;" : "=r"(res_hi) : "r"(a_hi), "r"(b_hi), "r"(c_hi));
    return ((uint64_t)res_hi << 32) | res_lo;
}

/* Byte-swap uint64 (big-endian counter <-> little-endian keccak lane) */
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    return ((uint64_t)__byte_perm(lo, 0u, 0x0123u) << 32)
         |  (uint64_t)__byte_perm(hi, 0u, 0x0123u);
}

/* Load 8 bytes from constant memory as little-endian uint64 */
__device__ __forceinline__ uint64_t load_le64_const(const uint8_t* p) {
    uint64_t v = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        v |= ((uint64_t)p[i]) << (i * 8);
    return v;
}

/* -- Difficulty comparison: first 32 output bytes (LE lanes 0-3) < target -- */
__device__ __forceinline__ bool hash_lt_difficulty(const uint64_t s0, const uint64_t s1, const uint64_t s2, const uint64_t s3) {
    // We compare word by word (big-endian).
    // ulonglong4 allows a vectorized load from constant memory.
    ulonglong4 diff = d_difficulty_be_vec;
    
    uint64_t h_be;
    
    h_be = bswap64(s0);
    if (h_be < diff.x) return true;
    if (h_be > diff.x) return false;

    h_be = bswap64(s1);
    if (h_be < diff.y) return true;
    if (h_be > diff.y) return false;

    h_be = bswap64(s2);
    if (h_be < diff.z) return true;
    if (h_be > diff.z) return false;

    h_be = bswap64(s3);
    if (h_be < diff.w) return true;
    if (h_be > diff.w) return false;

    return false;
}

/* -- Mining kernel --------------------------------------------------------- */

// Persistent kernel parameters
__device__ uint64_t g_nonce_allocator = 0;

#if WARP_COOP

template <int BLOCK_SIZE_T>
__global__
__launch_bounds__(BLOCK_SIZE_T, MIN_BLOCKS_PER_SM)
void mine_kernel(uint64_t        base_counter,
                 uint64_t        max_counter,
                 volatile int   *found_flag,
                 volatile uint64_t *found_counter)
{
    __shared__ volatile int s_found;
    if (threadIdx.x == 0) s_found = *found_flag;
    __syncthreads();

    int t = threadIdx.x % 32;
    int x = t % 5;
    int y = t / 5;
    
    // Precompute shuffle targets
    int src_t = t < 25 ? (((x + 3 * y) % 5) + 5 * x) : 0;
    int chi_t1 = t < 25 ? (((x + 1) % 5) + 5 * y) : 0;
    int chi_t2 = t < 25 ? (((x + 2) % 5) + 5 * y) : 0;
    int rot_val = t < 25 ? d_rot[t] : 0;
    
    uint64_t lane0 = load_le64_const(d_challenge +  0);
    uint64_t lane1 = load_le64_const(d_challenge +  8);
    uint64_t lane2 = load_le64_const(d_challenge + 16);
    uint64_t lane3 = load_le64_const(d_challenge + 24);
    uint64_t lane4 = load_le64_const(d_prefix    +  0);
    uint64_t lane5 = load_le64_const(d_prefix    +  8);
    uint64_t lane6 = load_le64_const(d_prefix    + 16);

    uint64_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint64_t stride = (gridDim.x * blockDim.x) / 32;

    for (uint64_t offset = warp_id; ; offset += stride) {
        uint64_t my_counter = base_counter + offset;

        if ((offset & 0x7F) == 0) {
            if (threadIdx.x == 0) s_found = *found_flag;
            __syncthreads();
        }
        if (s_found || my_counter >= max_counter) return;

        uint64_t st = 0;
        if (t == 0) st = lane0;
        else if (t == 1) st = lane1;
        else if (t == 2) st = lane2;
        else if (t == 3) st = lane3;
        else if (t == 4) st = lane4;
        else if (t == 5) st = lane5;
        else if (t == 6) st = lane6;
        else if (t == 7) st = bswap64(my_counter);
        else if (t == 8) st = 0x0000000000000001ULL;
        else if (t == 16) st = 0x8000000000000000ULL;

        #pragma unroll 2
        for (int r = 0; r < 24; r++) {
            /* Theta */
            uint64_t my_C = __shfl_sync(0xFFFFFFFF, st, x, 32) ^
                            __shfl_sync(0xFFFFFFFF, st, x + 5, 32) ^
                            __shfl_sync(0xFFFFFFFF, st, x + 10, 32) ^
                            __shfl_sync(0xFFFFFFFF, st, x + 15, 32) ^
                            __shfl_sync(0xFFFFFFFF, st, x + 20, 32);
            
            uint64_t Cx_minus_1 = __shfl_sync(0xFFFFFFFF, my_C, (x + 4) % 5, 32);
            uint64_t Cx_plus_1  = __shfl_sync(0xFFFFFFFF, my_C, (x + 1) % 5, 32);
            uint64_t D = Cx_minus_1 ^ ROTL64(Cx_plus_1, 1);
            st ^= D;

            /* Rho and Pi */
            uint64_t new_st = ROTL64(__shfl_sync(0xFFFFFFFF, st, src_t, 32), rot_val);
            st = new_st;

            /* Chi */
            uint64_t st_1 = __shfl_sync(0xFFFFFFFF, st, chi_t1, 32);
            uint64_t st_2 = __shfl_sync(0xFFFFFFFF, st, chi_t2, 32);
            st = st ^ (~st_1 & st_2);

            /* Iota */
            if (t == 0) {
                st ^= d_RC[r];
            }
        }

        uint64_t s0 = __shfl_sync(0xFFFFFFFF, st, 0, 32);
        uint64_t s1 = __shfl_sync(0xFFFFFFFF, st, 1, 32);
        uint64_t s2 = __shfl_sync(0xFFFFFFFF, st, 2, 32);
        uint64_t s3 = __shfl_sync(0xFFFFFFFF, st, 3, 32);
        
        if (t == 0) {
            if (hash_lt_difficulty(s0, s1, s2, s3)) {
                if (atomicCAS((int*)found_flag, 0, 1) == 0) {
                    *found_counter = my_counter;
                }
                s_found = 1;
            }
        }
    }
}

#else

template <int BLOCK_SIZE_T>
__global__
__launch_bounds__(BLOCK_SIZE_T, MIN_BLOCKS_PER_SM)
void mine_kernel(uint64_t        base_counter,
                 uint64_t        max_counter,
                 volatile int   *found_flag,
                 volatile uint64_t *found_counter)
{
    __shared__ volatile int s_found;
    if (threadIdx.x == 0) s_found = *found_flag;
    __syncthreads();

    /* Preload constant lanes 0-6 into registers (done once per thread) */
    uint64_t lane0 = load_le64_const(d_challenge +  0);
    uint64_t lane1 = load_le64_const(d_challenge +  8);
    uint64_t lane2 = load_le64_const(d_challenge + 16);
    uint64_t lane3 = load_le64_const(d_challenge + 24);
    uint64_t lane4 = load_le64_const(d_prefix    +  0);
    uint64_t lane5 = load_le64_const(d_prefix    +  8);
    uint64_t lane6 = load_le64_const(d_prefix    + 16);

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    // Strided nonce traversal
    for (uint64_t offset = tid; ; offset += stride) {
        uint64_t my_counter = base_counter + offset;
        
        // Every few iterations, check the global found flag.
        // We use a shared variable to reduce global memory pressure.
        if ((offset & 0x7F) == 0) {
            if (threadIdx.x == 0) s_found = *found_flag;
            __syncthreads();
        }
        if (s_found || my_counter >= max_counter) return;

        /*
         * Build keccak state from inputs. Scalar variables to avoid array indexing/spills.
         */
        uint64_t s00 = lane0;
        uint64_t s01 = lane1;
        uint64_t s02 = lane2;
        uint64_t s03 = lane3;
        uint64_t s04 = lane4;
        uint64_t s05 = lane5;
        uint64_t s06 = lane6;
        uint64_t s07 = bswap64(my_counter);
        uint64_t s08 = 0x0000000000000001ULL;
        uint64_t s09 = 0;
        uint64_t s10 = 0;
        uint64_t s11 = 0;
        uint64_t s12 = 0;
        uint64_t s13 = 0;
        uint64_t s14 = 0;
        uint64_t s15 = 0;
        uint64_t s16 = 0x8000000000000000ULL;
        uint64_t s17 = 0;
        uint64_t s18 = 0;
        uint64_t s19 = 0;
        uint64_t s20 = 0;
        uint64_t s21 = 0;
        uint64_t s22 = 0;
        uint64_t s23 = 0;
        uint64_t s24 = 0;

        #pragma unroll
        for (int r = 0; r < 24; r++) {
            /* Theta */
            uint64_t C0 = s00^s05^s10^s15^s20;
            uint64_t C1 = s01^s06^s11^s16^s21;
            uint64_t C2 = s02^s07^s12^s17^s22;
            uint64_t C3 = s03^s08^s13^s18^s23;
            uint64_t C4 = s04^s09^s14^s19^s24;

            uint64_t D0 = C4^ROTL64(C1, 1);
            uint64_t D1 = C0^ROTL64(C2, 1);
            uint64_t D2 = C1^ROTL64(C3, 1);
            uint64_t D3 = C2^ROTL64(C4, 1);
            uint64_t D4 = C3^ROTL64(C0, 1);

            s00^=D0; s05^=D0; s10^=D0; s15^=D0; s20^=D0;
            s01^=D1; s06^=D1; s11^=D1; s16^=D1; s21^=D1;
            s02^=D2; s07^=D2; s12^=D2; s17^=D2; s22^=D2;
            s03^=D3; s08^=D3; s13^=D3; s18^=D3; s23^=D3;
            s04^=D4; s09^=D4; s14^=D4; s19^=D4; s24^=D4;

            /* Rho and Pi */
            uint64_t t   = s01;
            s01 = ROTL64(s06, 44);
            s06 = ROTL64(s09, 20);
            s09 = ROTL64(s22, 61);
            s22 = ROTL64(s14, 39);
            s14 = ROTL64(s20, 18);
            s20 = ROTL64(s02, 62);
            s02 = ROTL64(s12, 43);
            s12 = ROTL64(s13, 25);
            s13 = ROTL64(s19,  8);
            s19 = ROTL64(s23, 56);
            s23 = ROTL64(s15, 41);
            s15 = ROTL64(s04, 27);
            s04 = ROTL64(s24, 14);
            s24 = ROTL64(s21,  2);
            s21 = ROTL64(s08, 55);
            s08 = ROTL64(s16, 45);
            s16 = ROTL64(s05, 36);
            s05 = ROTL64(s03, 28);
            s03 = ROTL64(s18, 21);
            s18 = ROTL64(s17, 15);
            s17 = ROTL64(s11, 10);
            s11 = ROTL64(s07,  6);
            s07 = ROTL64(s10,  3);
            s10 = ROTL64(t,    1);

            /* Chi (Optimized with LOP3 and correct temporaries) */
            uint64_t t0 = s00, t1 = s01;
            s00 = chi64_ptx(s00, s01, s02);
            s01 = chi64_ptx(s01, s02, s03);
            s02 = chi64_ptx(s02, s03, s04);
            s03 = chi64_ptx(s03, s04, t0);
            s04 = chi64_ptx(s04, t0, t1);

            t0 = s05; t1 = s06;
            s05 = chi64_ptx(s05, s06, s07);
            s06 = chi64_ptx(s06, s07, s08);
            s07 = chi64_ptx(s07, s08, s09);
            s08 = chi64_ptx(s08, s09, t0);
            s09 = chi64_ptx(s09, t0, t1);

            t0 = s10; t1 = s11;
            s10 = chi64_ptx(s10, s11, s12);
            s11 = chi64_ptx(s11, s12, s13);
            s12 = chi64_ptx(s12, s13, s14);
            s13 = chi64_ptx(s13, s14, t0);
            s14 = chi64_ptx(s14, t0, t1);

            t0 = s15; t1 = s16;
            s15 = chi64_ptx(s15, s16, s17);
            s16 = chi64_ptx(s16, s17, s18);
            s17 = chi64_ptx(s17, s18, s19);
            s18 = chi64_ptx(s18, s19, t0);
            s19 = chi64_ptx(s19, t0, t1);

            t0 = s20; t1 = s21;
            s20 = chi64_ptx(s20, s21, s22);
            s21 = chi64_ptx(s21, s22, s23);
            s22 = chi64_ptx(s22, s23, s24);
            s23 = chi64_ptx(s23, s24, t0);
            s24 = chi64_ptx(s24, t0, t1);

            /* Iota */
            s00 ^= d_RC[r];
        }

        if (hash_lt_difficulty(s00, s01, s02, s03)) {
            if (atomicCAS((int*)found_flag, 0, 1) == 0) {
                *found_counter = my_counter;
            }
            s_found = 1;
            return;
        }
    }
}
#endif

/* ============================== Host code ================================== */

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        char _buf[256]; \
        snprintf(_buf,sizeof(_buf),"%s failed: %s",#call,cudaGetErrorString(_e)); \
        fatal_json(_buf); \
    } \
} while(0)

static void json_escape(char *dst, size_t dsz, const char *src) {
    size_t di = 0;
    for (size_t i = 0; src[i] && di+4 < dsz; i++) {
        char c = src[i];
        if (c=='"'||c=='\\') { dst[di++]='\\'; dst[di++]=c; }
        else dst[di++]=c;
    }
    dst[di]=0;
}

static void fatal_json(const char *msg) {
    char esc[512]; json_escape(esc,sizeof(esc),msg);
    printf("{\"type\":\"error\",\"message\":\"%s\"}\n",esc);
    fflush(stdout); exit(2);
}

static int hexval(char c) {
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return -1;
}

static int parse_hex(const char *hex, uint8_t *out, size_t expected) {
    if (hex[0]=='0'&&(hex[1]=='x'||hex[1]=='X')) hex+=2;
    if (strlen(hex)!=expected*2) return -1;
    for (size_t i=0;i<expected;i++) {
        int hi=hexval(hex[i*2]), lo=hexval(hex[i*2+1]);
        if (hi<0||lo<0) return -1;
        out[i]=(uint8_t)((hi<<4)|lo);
    }
    return 0;
}

static uint64_t parse_u64(const char *s) {
    errno=0;
    uint64_t v=strtoull(s,NULL,10);
    if (errno) { fprintf(stderr,"bad integer: %s\n",s); exit(2); }
    return v;
}

#include <time.h>
static uint64_t now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts);
    return (uint64_t)ts.tv_sec*1000ULL + (uint64_t)ts.tv_nsec/1000000ULL;
}

/* Pack 32 difficulty bytes into 4 big-endian uint64 for fast word comparison */
static void pack_difficulty_be(const uint8_t *diff, uint64_t out[4]) {
    for (int i=0;i<4;i++) {
        out[i]=0;
        for (int j=0;j<8;j++) out[i]=(out[i]<<8)|diff[i*8+j];
    }
}

/* Autotuner Configs */
typedef void (*mine_kernel_t)(uint64_t, uint64_t, volatile int*, volatile uint64_t*);

struct KernelConfig {
    int block_size;
    mine_kernel_t func;
};

#define INSTANTIATE_KERNEL(BS) { BS, mine_kernel<BS> }

KernelConfig configs[] = {
    INSTANTIATE_KERNEL(64),
    INSTANTIATE_KERNEL(128),
    INSTANTIATE_KERNEL(192),
    INSTANTIATE_KERNEL(256),
#if WARP_COOP
    INSTANTIATE_KERNEL(512)
#endif
};

static double test_hashrate(KernelConfig cfg, int wpt, cudaStream_t stream, int* d_flag, uint64_t* d_counter) {
    uint64_t test_hashes = 20000000ULL; // 20M hashes for the test
    
#if WARP_COOP
    size_t grid = (test_hashes / wpt) / (cfg.block_size / 32);
#else
    size_t grid = test_hashes / (cfg.block_size * wpt);
#endif
    if (grid == 0) grid = 1;

#if WARP_COOP
    uint64_t actual_hashes = (uint64_t)grid * (cfg.block_size / 32) * wpt;
#else
    uint64_t actual_hashes = (uint64_t)grid * cfg.block_size * wpt;
#endif

    CUDA_CHECK(cudaMemsetAsync(d_flag, 0, sizeof(int), stream));
    
    // Warmup
    cfg.func<<<(unsigned)grid, cfg.block_size, 0, stream>>>(0, actual_hashes, d_flag, d_counter);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Benchmark
    CUDA_CHECK(cudaMemsetAsync(d_flag, 0, sizeof(int), stream));
    uint64_t t_start = now_ms();
    
    // Launch a few times to average out overhead
    for(int i=0; i<3; i++) {
        cfg.func<<<(unsigned)grid, cfg.block_size, 0, stream>>>(0, actual_hashes, d_flag, d_counter);
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    uint64_t t_end = now_ms();
    
    double elapsed_sec = (t_end - t_start) / 1000.0;
    if (elapsed_sec <= 0) return 0;
    return ((actual_hashes * 3) / elapsed_sec) / 1000000.0; // return MH/s
}

int main(int argc, char **argv) {
    const char *challenge_hex    = NULL;
    const char *difficulty_hex   = NULL;
    const char *nonce_prefix_hex = NULL;
    uint64_t    start_counter    = 0;
    size_t      batch_size       = 0;   /* 0 = auto */
    int         device_index     = 0;
    uint64_t    progress_ms      = 2000;
    int         list_devices     = 0;

    for (int i=1;i<argc;i++) {
#define NEXTARG (i+1<argc?argv[++i]:(fprintf(stderr,"missing arg after %s\n",argv[i]),exit(2),(char*)0))
        if      (!strcmp(argv[i],"--challenge"))     challenge_hex    = NEXTARG;
        else if (!strcmp(argv[i],"--difficulty"))    difficulty_hex   = NEXTARG;
        else if (!strcmp(argv[i],"--nonce-prefix"))  nonce_prefix_hex = NEXTARG;
        else if (!strcmp(argv[i],"--start"))         start_counter    = parse_u64(NEXTARG);
        else if (!strcmp(argv[i],"--batch-size"))    batch_size       = (size_t)parse_u64(NEXTARG);
        else if (!strcmp(argv[i],"--device-index"))  device_index     = (int)parse_u64(NEXTARG);
        else if (!strcmp(argv[i],"--platform-index")) NEXTARG; /* ignored; CUDA has no platforms */
        else if (!strcmp(argv[i],"--progress-ms"))   progress_ms      = parse_u64(NEXTARG);
        else if (!strcmp(argv[i],"--local-size"))    NEXTARG;          /* ignored */
        else if (!strcmp(argv[i],"--list-devices"))  list_devices     = 1;
#undef NEXTARG
    }

    /* Enumerate / list devices */
    int num_devs=0;
    if (cudaGetDeviceCount(&num_devs)!=cudaSuccess||num_devs==0)
        fatal_json("No CUDA devices found. Install NVIDIA driver >= 520.");

    if (list_devices) {
        printf("[");
        for (int d=0;d<num_devs;d++) {
            cudaDeviceProp p; cudaGetDeviceProperties(&p,d);
            char dn[512]={0};
            json_escape(dn,sizeof(dn),p.name);
            if (d) printf(",");
            printf("{\"platform_index\":0,\"device_index\":%d,"
                   "\"platform\":\"CUDA\",\"device\":\"%s\","
                   "\"vendor\":\"NVIDIA\",\"cu\":%d,\"max_wg\":%zu}",
                   d,dn,p.multiProcessorCount,(size_t)p.maxThreadsPerBlock);
        }
        printf("]\n"); fflush(stdout); return 0;
    }

    /* Validate required args */
    if (!challenge_hex||!difficulty_hex||!nonce_prefix_hex) {
        fprintf(stderr,"usage: gpu_miner_cuda --challenge HEX64 --difficulty HEX64 "
                       "--nonce-prefix HEX48 --start N\n");
        return 2;
    }
    uint8_t challenge[32], difficulty[32], nonce_prefix[24];
    if (parse_hex(challenge_hex,    challenge,    32)) fatal_json("invalid --challenge hex");
    if (parse_hex(difficulty_hex,   difficulty,   32)) fatal_json("invalid --difficulty hex");
    if (parse_hex(nonce_prefix_hex, nonce_prefix, 24)) fatal_json("invalid --nonce-prefix hex");

    /* Select device */
    if (device_index>=num_devs) fatal_json("--device-index out of range");
    CUDA_CHECK(cudaSetDevice(device_index));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,device_index));

    /* Upload constants to device */
    CUDA_CHECK(cudaMemcpyToSymbol(d_challenge, challenge,    32));
    CUDA_CHECK(cudaMemcpyToSymbol(d_prefix,    nonce_prefix, 24));
    uint64_t diff_be[4]; pack_difficulty_be(difficulty,diff_be);
    // Copy as ulonglong4
    CUDA_CHECK(cudaMemcpyToSymbol(d_difficulty_be_vec, diff_be,  32));

    /* Allocate device-side output buffers */
    int      *d_flag;    CUDA_CHECK(cudaMalloc(&d_flag,    sizeof(int)));
    uint64_t *d_counter; CUDA_CHECK(cudaMalloc(&d_counter, sizeof(uint64_t)));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* Auto-Tune block size and work per thread */
    KernelConfig best_cfg = configs[1]; // default 128
    int best_wpt = 16;
    double best_mhs = 0;

    if (batch_size == 0) {
        printf("{\"type\":\"progress\", \"message\":\"Starting Auto-Tune (Block Size & WPT)...\"}\n");
        fflush(stdout);
        
        // Backup difficulty and set to zero to avoid early exit during benchmark
        ulonglong4 zero_diff = {0,0,0,0};
        CUDA_CHECK(cudaMemcpyToSymbol(d_difficulty_be_vec, &zero_diff, 32));

        int wpt_opts[] = {1, 2, 4, 8, 16, 32};
        for (int i=0; i < sizeof(configs)/sizeof(configs[0]); i++) {
            for (int w=0; w < 6; w++) {
                int wpt = wpt_opts[w];
                double mhs = test_hashrate(configs[i], wpt, stream, d_flag, d_counter);
                printf("{\"type\":\"autotune\", \"block_size\":%d, \"work_per_thread\":%d, \"mhs\":%.2f}\n", 
                    configs[i].block_size, wpt, mhs);
                fflush(stdout);
                if (mhs > best_mhs) {
                    best_mhs = mhs;
                    best_cfg = configs[i];
                    best_wpt = wpt;
                }
            }
        }
        
        // Restore difficulty
        CUDA_CHECK(cudaMemcpyToSymbol(d_difficulty_be_vec, diff_be, 32));

        printf("{\"type\":\"autotune_done\", \"best_block_size\":%d, \"best_wpt\":%d, \"best_mhs\":%.2f}\n", 
                best_cfg.block_size, best_wpt, best_mhs);
        fflush(stdout);
        
        // Target ~50ms batch to keep progress reporting responsive
        batch_size = (size_t)(best_mhs * 1000000.0 * 0.05);
        if (batch_size < 1000000) batch_size = 1000000;
    }

    /* Report device info */
    {
        char dn[512]={0};
        json_escape(dn,sizeof(dn),prop.name);
        printf("{\"type\":\"device\",\"platform\":\"CUDA\","
               "\"device\":\"%s\",\"vendor\":\"NVIDIA\","
               "\"cu\":%d,\"max_wg\":%zu,\"batch_size\":%zu,\"local_size\":%d,\"wpt\":%d}\n",
               dn,prop.multiProcessorCount,
               (size_t)prop.maxThreadsPerBlock,batch_size,best_cfg.block_size,best_wpt);
        fflush(stdout);
    }

    /* Mining loop */
    uint64_t next_counter = start_counter;
    uint64_t total_hashes = 0;
    uint64_t last_prog_t  = now_ms();
    
    size_t grid;
#if WARP_COOP
    grid = (batch_size / best_wpt) / (best_cfg.block_size / 32);
#else
    grid = batch_size / ((size_t)best_cfg.block_size * best_wpt);
#endif
    if (grid == 0) grid = 1;

    // Use a persistent kernel. We still launch in a loop so the host can periodically print progress.
    // The kernel itself will run for `grid * BLOCK_SIZE * WORK_PER_THREAD` nonces per launch,
    // but the threads inside it will strided-loop over that range.
    
    int *h_flag;
    uint64_t *h_counter;
    CUDA_CHECK(cudaMallocHost(&h_flag, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_counter, sizeof(uint64_t)));

    for (;;) {
        *h_flag = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice, stream));

#if WARP_COOP
        uint64_t hashes_this_iter = (uint64_t)grid * (best_cfg.block_size / 32) * best_wpt;
#else
        uint64_t hashes_this_iter = (uint64_t)grid * best_cfg.block_size * best_wpt;
#endif
        uint64_t max_counter = next_counter + hashes_this_iter;

        best_cfg.func<<<(unsigned)grid, best_cfg.block_size, 0, stream>>>(
            next_counter, max_counter, d_flag, d_counter);
        CUDA_CHECK(cudaGetLastError());
        
        // Asynchronous memcpy to pinned host memory
        CUDA_CHECK(cudaMemcpyAsync(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        total_hashes += hashes_this_iter;
        next_counter = max_counter;

        if (*h_flag) {
            uint64_t winner=0;
            CUDA_CHECK(cudaMemcpy(&winner, d_counter, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            printf("{\"type\":\"found\",\"counter\":\"%" PRIu64 "\",\"hashes\":\"%" PRIu64 "\"}\n",
                   winner, total_hashes);
            fflush(stdout);
            break;
        }

        uint64_t now=now_ms();
        if (now-last_prog_t>=progress_ms) {
            printf("{\"type\":\"progress\",\"hashes\":\"%" PRIu64 "\",\"nonce\":\"%" PRIu64 "\"}\n",
                   total_hashes, next_counter);
            fflush(stdout);
            last_prog_t=now;
        }
    }

    cudaFreeHost(h_flag);
    cudaFreeHost(h_counter);
    cudaStreamDestroy(stream);
    cudaFree(d_flag);
    cudaFree(d_counter);
    return 0;
}

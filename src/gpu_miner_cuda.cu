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
#define BLOCK_SIZE        256   /* threads per block -- multiple of 32 */
#define MIN_BLOCKS_PER_SM   2   /* hint to __launch_bounds__ for reg alloc */
#define WORK_PER_THREAD     8   /* nonces each thread tries per kernel call */

/* -- Constant memory (L1-cached, broadcast to all threads in warp) --------- */
__constant__ uint8_t  d_challenge[32];
__constant__ uint8_t  d_prefix[24];
__constant__ uint64_t d_difficulty_be[4]; /* difficulty as 4 big-endian u64 */
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
#define ROTL64(x,n) (((x)<<(n))|((x)>>(64-(n))))

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

/* -- Keccak-f[1600] --------------------------------------------------------
 *
 * rho+pi are applied via an explicit B[] array, matching keccak256_mine.cl
 * exactly. This avoids any aliasing ambiguity in the in-place chained form.
 * The 25 B variables and 25 s variables all live in registers (no spill at
 * BLOCK_SIZE=256, MIN_BLOCKS_PER_SM=2 with nvcc -O3).
 */
__device__ __forceinline__ void keccak_f1600(uint64_t s[25]) {
    uint64_t C0,C1,C2,C3,C4, D0,D1,D2,D3,D4;
    uint64_t B00,B01,B02,B03,B04,B05,B06,B07,B08,B09,
             B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,
             B20,B21,B22,B23,B24;

    #pragma unroll
    for (int r = 0; r < 24; r++) {
        /* theta */
        C0=s[0]^s[5]^s[10]^s[15]^s[20]; C1=s[1]^s[6]^s[11]^s[16]^s[21];
        C2=s[2]^s[7]^s[12]^s[17]^s[22]; C3=s[3]^s[8]^s[13]^s[18]^s[23];
        C4=s[4]^s[9]^s[14]^s[19]^s[24];
        D0=C4^ROTL64(C1,1); D1=C0^ROTL64(C2,1); D2=C1^ROTL64(C3,1);
        D3=C2^ROTL64(C4,1); D4=C3^ROTL64(C0,1);
        s[ 0]^=D0; s[ 5]^=D0; s[10]^=D0; s[15]^=D0; s[20]^=D0;
        s[ 1]^=D1; s[ 6]^=D1; s[11]^=D1; s[16]^=D1; s[21]^=D1;
        s[ 2]^=D2; s[ 7]^=D2; s[12]^=D2; s[17]^=D2; s[22]^=D2;
        s[ 3]^=D3; s[ 8]^=D3; s[13]^=D3; s[18]^=D3; s[23]^=D3;
        s[ 4]^=D4; s[ 9]^=D4; s[14]^=D4; s[19]^=D4; s[24]^=D4;

        /* rho + pi -> B[]  (verified against keccak256_mine.cl) */
        B00=           s[ 0]; B10=ROTL64(s[ 1], 1); B20=ROTL64(s[ 2],62);
        B05=ROTL64(s[ 3],28); B15=ROTL64(s[ 4],27);
        B16=ROTL64(s[ 5],36); B01=ROTL64(s[ 6],44); B11=ROTL64(s[ 7], 6);
        B21=ROTL64(s[ 8],55); B06=ROTL64(s[ 9],20);
        B07=ROTL64(s[10], 3); B17=ROTL64(s[11],10); B02=ROTL64(s[12],43);
        B12=ROTL64(s[13],25); B22=ROTL64(s[14],39);
        B23=ROTL64(s[15],41); B08=ROTL64(s[16],45); B18=ROTL64(s[17],15);
        B03=ROTL64(s[18],21); B13=ROTL64(s[19], 8);
        B14=ROTL64(s[20],18); B24=ROTL64(s[21], 2); B09=ROTL64(s[22],61);
        B19=ROTL64(s[23],56); B04=ROTL64(s[24],14);

        /* chi */
        s[ 0]=B00^(~B01&B02); s[ 1]=B01^(~B02&B03); s[ 2]=B02^(~B03&B04);
        s[ 3]=B03^(~B04&B00); s[ 4]=B04^(~B00&B01);
        s[ 5]=B05^(~B06&B07); s[ 6]=B06^(~B07&B08); s[ 7]=B07^(~B08&B09);
        s[ 8]=B08^(~B09&B05); s[ 9]=B09^(~B05&B06);
        s[10]=B10^(~B11&B12); s[11]=B11^(~B12&B13); s[12]=B12^(~B13&B14);
        s[13]=B13^(~B14&B10); s[14]=B14^(~B10&B11);
        s[15]=B15^(~B16&B17); s[16]=B16^(~B17&B18); s[17]=B17^(~B18&B19);
        s[18]=B18^(~B19&B15); s[19]=B19^(~B15&B16);
        s[20]=B20^(~B21&B22); s[21]=B21^(~B22&B23); s[22]=B22^(~B23&B24);
        s[23]=B23^(~B24&B20); s[24]=B24^(~B20&B21);

        /* iota */
        s[0] ^= d_RC[r];
    }
}

/* -- Difficulty comparison: first 32 output bytes (LE lanes 0-3) < target -- */
__device__ __forceinline__ bool hash_lt_difficulty(const uint64_t s[25]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t h_be = bswap64(s[i]);
        if (h_be < d_difficulty_be[i]) return true;
        if (h_be > d_difficulty_be[i]) return false;
    }
    return false;
}

/* -- Mining kernel --------------------------------------------------------- */
__global__
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
void mine_kernel(uint64_t        base_counter,
                 volatile int   *found_flag,
                 volatile uint64_t *found_counter)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t my_counter = base_counter + tid * (uint64_t)WORK_PER_THREAD;

    /* Preload constant lanes 0-6 into registers (done once per thread) */
    uint64_t lane0 = load_le64_const(d_challenge +  0);
    uint64_t lane1 = load_le64_const(d_challenge +  8);
    uint64_t lane2 = load_le64_const(d_challenge + 16);
    uint64_t lane3 = load_le64_const(d_challenge + 24);
    uint64_t lane4 = load_le64_const(d_prefix    +  0);
    uint64_t lane5 = load_le64_const(d_prefix    +  8);
    uint64_t lane6 = load_le64_const(d_prefix    + 16);

    #pragma unroll
    for (int w = 0; w < WORK_PER_THREAD; w++) {
        if (*found_flag) return;  /* early exit if another thread won */

        /*
         * Build keccak state from inputs.
         * Initial state = all-zero XOR input lanes (XOR with 0 = assign).
         * Input: challenge[32] || prefix[24] || counter_be[8] = 64 bytes
         *   -> lanes 0-7.  Padding: lane[8] ^= 0x01, lane[16] ^= 0x80..0
         */
        uint64_t s[25];
        s[ 0]=lane0; s[ 1]=lane1; s[ 2]=lane2; s[ 3]=lane3;
        s[ 4]=lane4; s[ 5]=lane5; s[ 6]=lane6;
        s[ 7]=bswap64(my_counter + (uint64_t)w); /* big-endian counter */
        s[ 8]=0x0000000000000001ULL;             /* keccak padding byte */
        s[ 9]=0; s[10]=0; s[11]=0; s[12]=0;
        s[13]=0; s[14]=0; s[15]=0;
        s[16]=0x8000000000000000ULL;             /* last byte of rate */
        s[17]=0; s[18]=0; s[19]=0; s[20]=0;
        s[21]=0; s[22]=0; s[23]=0; s[24]=0;

        keccak_f1600(s);

        if (hash_lt_difficulty(s)) {
            if (atomicCAS((int*)found_flag, 0, 1) == 0)
                *found_counter = my_counter + (uint64_t)w;
            return;
        }
    }
}

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

    /* Auto batch size: target ~20 ms per launch, scale with SM count */
    if (batch_size==0) {
        size_t auto_b=(size_t)prop.multiProcessorCount*2048*WORK_PER_THREAD;
        if (auto_b < (1u<<20)) auto_b = (1u<<20);  /* min 1M  */
        if (auto_b > (1u<<25)) auto_b = (1u<<25);  /* max 32M */
        size_t grain = (size_t)BLOCK_SIZE*WORK_PER_THREAD;
        batch_size = ((auto_b+grain-1)/grain)*grain;
    }

    /* Upload constants to device */
    CUDA_CHECK(cudaMemcpyToSymbol(d_challenge, challenge,    32));
    CUDA_CHECK(cudaMemcpyToSymbol(d_prefix,    nonce_prefix, 24));
    uint64_t diff_be[4]; pack_difficulty_be(difficulty,diff_be);
    CUDA_CHECK(cudaMemcpyToSymbol(d_difficulty_be, diff_be,  32));

    /* Allocate device-side output buffers */
    int      *d_flag;    CUDA_CHECK(cudaMalloc(&d_flag,    sizeof(int)));
    uint64_t *d_counter; CUDA_CHECK(cudaMalloc(&d_counter, sizeof(uint64_t)));

    /* Report device info */
    {
        char dn[512]={0};
        json_escape(dn,sizeof(dn),prop.name);
        printf("{\"type\":\"device\",\"platform\":\"CUDA\","
               "\"device\":\"%s\",\"vendor\":\"NVIDIA\","
               "\"cu\":%d,\"max_wg\":%zu,\"batch_size\":%zu,\"local_size\":%d}\n",
               dn,prop.multiProcessorCount,
               (size_t)prop.maxThreadsPerBlock,batch_size,BLOCK_SIZE);
        fflush(stdout);
    }

    /* Mining loop */
    uint64_t next_counter = start_counter;
    uint64_t total_hashes = 0;
    uint64_t last_prog_t  = now_ms();
    size_t   grid         = batch_size / ((size_t)BLOCK_SIZE * WORK_PER_THREAD);
    if (grid==0) grid=1;

    for (;;) {
        CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));

        mine_kernel<<<(unsigned)grid, BLOCK_SIZE>>>(
            next_counter, d_flag, d_counter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        total_hashes += (uint64_t)grid * BLOCK_SIZE * WORK_PER_THREAD;
        next_counter += (uint64_t)grid * BLOCK_SIZE * WORK_PER_THREAD;

        int flag=0;
        CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (flag) {
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

    cudaFree(d_flag);
    cudaFree(d_counter);
    return 0;
}

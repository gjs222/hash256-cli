/*
 * gpu_miner.c  –  hash256 keccak256 GPU miner
 *
 * Dynamically loads OpenCL at runtime (no SDK headers required).
 * Compile with:  gcc -O2 gpu_miner.c -o gpu_miner.exe          (Windows/MSYS2)
 *                gcc -O2 gpu_miner.c -ldl -o gpu_miner          (Linux)
 *
 * Usage:
 *   gpu_miner.exe --challenge HEX64 --difficulty HEX64 \
 *                 --nonce-prefix HEX48 --start N      \
 *                 [--batch-size N] [--local-size N]   \
 *                 [--platform-index N] [--device-index N] \
 *                 [--progress-ms N] [--list-devices]
 *
 * Output: newline-delimited JSON on stdout.
 *   {"type":"device",   "platform":"...", "device":"...", "cu":N, "max_wg":N}
 *   {"type":"progress", "hashes":"N", "nonce":"N"}
 *   {"type":"found",    "counter":"N", "hashes":"N"}
 *   {"type":"error",    "message":"..."}
 */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <time.h>
#endif

/* ── OpenCL type definitions (no SDK needed) ──────────────────────────────── */
typedef intptr_t  cl_intptr_t;
typedef uintptr_t cl_uintptr_t;
typedef cl_uintptr_t cl_bitfield;
typedef cl_bitfield  cl_device_type;
typedef cl_bitfield  cl_mem_flags;
typedef cl_bitfield  cl_command_queue_properties;
typedef uint32_t  cl_uint;
typedef int32_t   cl_int;
typedef uint64_t  cl_ulong;
typedef uint8_t   cl_bool;
typedef uint16_t  cl_ushort;
typedef size_t    cl_size_t;

typedef struct _cl_platform_id    *cl_platform_id;
typedef struct _cl_device_id      *cl_device_id;
typedef struct _cl_context        *cl_context;
typedef struct _cl_command_queue  *cl_command_queue;
typedef struct _cl_mem            *cl_mem;
typedef struct _cl_program        *cl_program;
typedef struct _cl_kernel         *cl_kernel;
typedef struct _cl_event          *cl_event;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_kernel_work_group_info;

#define CL_SUCCESS                0
#define CL_FALSE                  0
#define CL_TRUE                   1
#define CL_DEVICE_TYPE_DEFAULT    (1u << 0)
#define CL_DEVICE_TYPE_CPU        (1u << 1)
#define CL_DEVICE_TYPE_GPU        (1u << 2)
#define CL_DEVICE_TYPE_ACCELERATOR (1u << 3)
#define CL_DEVICE_TYPE_ALL        0xFFFFFFFFu
#define CL_MEM_READ_ONLY          (1u << 2)
#define CL_MEM_WRITE_ONLY         (1u << 1)
#define CL_MEM_COPY_HOST_PTR      (1u << 5)
#define CL_PROGRAM_BUILD_LOG      0x1183
#define CL_DEVICE_NAME            0x102B
#define CL_DEVICE_VENDOR          0x102C
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_PLATFORM_NAME          0x0902

typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*PFN_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (*PFN_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
typedef cl_context (*PFN_clCreateContext)(const cl_intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
typedef cl_command_queue (*PFN_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
typedef cl_mem (*PFN_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_program (*PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (*PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
typedef cl_int (*PFN_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
typedef cl_kernel (*PFN_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (*PFN_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*PFN_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*PFN_clFinish)(cl_command_queue);
typedef cl_int (*PFN_clReleaseMemObject)(cl_mem);
typedef cl_int (*PFN_clReleaseKernel)(cl_kernel);
typedef cl_int (*PFN_clReleaseProgram)(cl_program);
typedef cl_int (*PFN_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*PFN_clReleaseContext)(cl_context);

typedef struct {
    PFN_clGetPlatformIDs          clGetPlatformIDs;
    PFN_clGetPlatformInfo         clGetPlatformInfo;
    PFN_clGetDeviceIDs            clGetDeviceIDs;
    PFN_clGetDeviceInfo           clGetDeviceInfo;
    PFN_clCreateContext           clCreateContext;
    PFN_clCreateCommandQueue      clCreateCommandQueue;
    PFN_clCreateBuffer            clCreateBuffer;
    PFN_clCreateProgramWithSource clCreateProgramWithSource;
    PFN_clBuildProgram            clBuildProgram;
    PFN_clGetProgramBuildInfo     clGetProgramBuildInfo;
    PFN_clCreateKernel            clCreateKernel;
    PFN_clSetKernelArg            clSetKernelArg;
    PFN_clEnqueueWriteBuffer      clEnqueueWriteBuffer;
    PFN_clEnqueueReadBuffer       clEnqueueReadBuffer;
    PFN_clEnqueueNDRangeKernel    clEnqueueNDRangeKernel;
    PFN_clFinish                  clFinish;
    PFN_clReleaseMemObject        clReleaseMemObject;
    PFN_clReleaseKernel           clReleaseKernel;
    PFN_clReleaseProgram          clReleaseProgram;
    PFN_clReleaseCommandQueue     clReleaseCommandQueue;
    PFN_clReleaseContext          clReleaseContext;
} cl_api;

static cl_api cl;

/* ── Timer ──────────────────────────────────────────────────────────────── */
#ifdef _WIN32
static uint64_t now_ms(void) {
    FILETIME ft;
    ULARGE_INTEGER uli;
    GetSystemTimeAsFileTime(&ft);
    uli.LowPart  = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return (uli.QuadPart / 10000ull) - 11644473600000ull;
}
#else
static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}
#endif

/* ── OpenCL dynamic loader ───────────────────────────────────────────────── */
static void *opencl_sym(void *lib, const char *name) {
#ifdef _WIN32
    void *sym = (void *)GetProcAddress((HMODULE)lib, name);
#else
    void *sym = dlsym(lib, name);
#endif
    if (!sym) { fprintf(stderr, "missing OpenCL symbol: %s\n", name); exit(2); }
    return sym;
}

static int load_opencl(void) {
#ifdef _WIN32
    void *lib = LoadLibraryA("OpenCL.dll");
#else
    void *lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) lib = dlopen("libOpenCL.so.1", RTLD_NOW);
#endif
    if (!lib) return 0;
    cl.clGetPlatformIDs          = (PFN_clGetPlatformIDs)         opencl_sym(lib, "clGetPlatformIDs");
    cl.clGetPlatformInfo         = (PFN_clGetPlatformInfo)        opencl_sym(lib, "clGetPlatformInfo");
    cl.clGetDeviceIDs            = (PFN_clGetDeviceIDs)           opencl_sym(lib, "clGetDeviceIDs");
    cl.clGetDeviceInfo           = (PFN_clGetDeviceInfo)          opencl_sym(lib, "clGetDeviceInfo");
    cl.clCreateContext           = (PFN_clCreateContext)          opencl_sym(lib, "clCreateContext");
    cl.clCreateCommandQueue      = (PFN_clCreateCommandQueue)     opencl_sym(lib, "clCreateCommandQueue");
    cl.clCreateBuffer            = (PFN_clCreateBuffer)           opencl_sym(lib, "clCreateBuffer");
    cl.clCreateProgramWithSource = (PFN_clCreateProgramWithSource)opencl_sym(lib, "clCreateProgramWithSource");
    cl.clBuildProgram            = (PFN_clBuildProgram)           opencl_sym(lib, "clBuildProgram");
    cl.clGetProgramBuildInfo     = (PFN_clGetProgramBuildInfo)    opencl_sym(lib, "clGetProgramBuildInfo");
    cl.clCreateKernel            = (PFN_clCreateKernel)           opencl_sym(lib, "clCreateKernel");
    cl.clSetKernelArg            = (PFN_clSetKernelArg)           opencl_sym(lib, "clSetKernelArg");
    cl.clEnqueueWriteBuffer      = (PFN_clEnqueueWriteBuffer)     opencl_sym(lib, "clEnqueueWriteBuffer");
    cl.clEnqueueReadBuffer       = (PFN_clEnqueueReadBuffer)      opencl_sym(lib, "clEnqueueReadBuffer");
    cl.clEnqueueNDRangeKernel    = (PFN_clEnqueueNDRangeKernel)  opencl_sym(lib, "clEnqueueNDRangeKernel");
    cl.clFinish                  = (PFN_clFinish)                 opencl_sym(lib, "clFinish");
    cl.clReleaseMemObject        = (PFN_clReleaseMemObject)       opencl_sym(lib, "clReleaseMemObject");
    cl.clReleaseKernel           = (PFN_clReleaseKernel)          opencl_sym(lib, "clReleaseKernel");
    cl.clReleaseProgram          = (PFN_clReleaseProgram)         opencl_sym(lib, "clReleaseProgram");
    cl.clReleaseCommandQueue     = (PFN_clReleaseCommandQueue)    opencl_sym(lib, "clReleaseCommandQueue");
    cl.clReleaseContext          = (PFN_clReleaseContext)         opencl_sym(lib, "clReleaseContext");
    return 1;
}

/* ── Hex helpers ─────────────────────────────────────────────────────────── */
static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int parse_hex(const char *hex, uint8_t *out, size_t expected) {
    if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) hex += 2;
    size_t n = strlen(hex);
    if (n != expected * 2) return -1;
    for (size_t i = 0; i < expected; i++) {
        int hi = hexval(hex[i * 2]), lo = hexval(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return 0;
}
static uint64_t parse_u64(const char *s) {
    errno = 0;
    uint64_t v = strtoull(s, NULL, 10);
    if (errno) { fprintf(stderr, "bad integer: %s\n", s); exit(2); }
    return v;
}

/* ── JSON helpers ────────────────────────────────────────────────────────── */
static void json_escape(char *dst, size_t dsz, const char *src) {
    size_t di = 0;
    for (size_t i = 0; src[i] && di + 4 < dsz; i++) {
        char c = src[i];
        if (c == '"' || c == '\\') { dst[di++] = '\\'; dst[di++] = c; }
        else dst[di++] = c;
    }
    dst[di] = 0;
}

/* ── Device info helpers ─────────────────────────────────────────────────── */
static void get_str_platform(cl_platform_id p, cl_platform_info what, char *out, size_t n) {
    if (cl.clGetPlatformInfo(p, what, n, out, NULL) != CL_SUCCESS) snprintf(out, n, "?");
}
static void get_str_device(cl_device_id d, cl_device_info what, char *out, size_t n) {
    if (cl.clGetDeviceInfo(d, what, n, out, NULL) != CL_SUCCESS) snprintf(out, n, "?");
}

/* ── Embedded OpenCL kernel ──────────────────────────────────────────────── */
/*
 * keccak256( challenge[32] || nonce_prefix[24] || counter_be[8] ) < difficulty
 * Uses original Keccak padding (0x01 … 0x80), rate = 136 bytes.
 */
static const char *KERNEL_SRC =
"__constant ulong RC[24] = {\n"
"  0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,\n"
"  0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,\n"
"  0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,\n"
"  0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,\n"
"  0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,\n"
"  0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,\n"
"  0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,\n"
"  0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL\n"
"};\n"
"#define ROTL64(x,n) (((x)<<(n))|((x)>>(64u-(n))))\n"
"void keccak_f1600(ulong *st) {\n"
"  ulong C0,C1,C2,C3,C4,D0,D1,D2,D3,D4;\n"
"  ulong B00,B01,B02,B03,B04,B05,B06,B07,B08,B09;\n"
"  ulong B10,B11,B12,B13,B14,B15,B16,B17,B18,B19;\n"
"  ulong B20,B21,B22,B23,B24;\n"
"  for(int r=0;r<24;r++){\n"
"    C0=st[0]^st[5]^st[10]^st[15]^st[20];\n"
"    C1=st[1]^st[6]^st[11]^st[16]^st[21];\n"
"    C2=st[2]^st[7]^st[12]^st[17]^st[22];\n"
"    C3=st[3]^st[8]^st[13]^st[18]^st[23];\n"
"    C4=st[4]^st[9]^st[14]^st[19]^st[24];\n"
"    D0=C4^ROTL64(C1,1); D1=C0^ROTL64(C2,1); D2=C1^ROTL64(C3,1);\n"
"    D3=C2^ROTL64(C4,1); D4=C3^ROTL64(C0,1);\n"
"    st[0]^=D0;st[5]^=D0;st[10]^=D0;st[15]^=D0;st[20]^=D0;\n"
"    st[1]^=D1;st[6]^=D1;st[11]^=D1;st[16]^=D1;st[21]^=D1;\n"
"    st[2]^=D2;st[7]^=D2;st[12]^=D2;st[17]^=D2;st[22]^=D2;\n"
"    st[3]^=D3;st[8]^=D3;st[13]^=D3;st[18]^=D3;st[23]^=D3;\n"
"    st[4]^=D4;st[9]^=D4;st[14]^=D4;st[19]^=D4;st[24]^=D4;\n"
"    B00=           st[0];\n"
"    B10=ROTL64(st[1], 1); B20=ROTL64(st[2],62);\n"
"    B05=ROTL64(st[3],28); B15=ROTL64(st[4],27);\n"
"    B16=ROTL64(st[5],36); B01=ROTL64(st[6],44);\n"
"    B11=ROTL64(st[7], 6); B21=ROTL64(st[8],55);\n"
"    B06=ROTL64(st[9],20); B07=ROTL64(st[10],3);\n"
"    B17=ROTL64(st[11],10);B02=ROTL64(st[12],43);\n"
"    B12=ROTL64(st[13],25);B22=ROTL64(st[14],39);\n"
"    B23=ROTL64(st[15],41);B08=ROTL64(st[16],45);\n"
"    B18=ROTL64(st[17],15);B03=ROTL64(st[18],21);\n"
"    B13=ROTL64(st[19], 8);B14=ROTL64(st[20],18);\n"
"    B24=ROTL64(st[21], 2);B09=ROTL64(st[22],61);\n"
"    B19=ROTL64(st[23],56);B04=ROTL64(st[24],14);\n"
"    st[0]=B00^(~B01&B02);st[1]=B01^(~B02&B03);\n"
"    st[2]=B02^(~B03&B04);st[3]=B03^(~B04&B00);\n"
"    st[4]=B04^(~B00&B01);\n"
"    st[5]=B05^(~B06&B07);st[6]=B06^(~B07&B08);\n"
"    st[7]=B07^(~B08&B09);st[8]=B08^(~B09&B05);\n"
"    st[9]=B09^(~B05&B06);\n"
"    st[10]=B10^(~B11&B12);st[11]=B11^(~B12&B13);\n"
"    st[12]=B12^(~B13&B14);st[13]=B13^(~B14&B10);\n"
"    st[14]=B14^(~B10&B11);\n"
"    st[15]=B15^(~B16&B17);st[16]=B16^(~B17&B18);\n"
"    st[17]=B17^(~B18&B19);st[18]=B18^(~B19&B15);\n"
"    st[19]=B19^(~B15&B16);\n"
"    st[20]=B20^(~B21&B22);st[21]=B21^(~B22&B23);\n"
"    st[22]=B22^(~B23&B24);st[23]=B23^(~B24&B20);\n"
"    st[24]=B24^(~B20&B21);\n"
"    st[0]^=RC[r];\n"
"  }\n"
"}\n"
"void keccak256_64(const uchar *in, uchar *hash) {\n"
"  ulong st[25];\n"
"  for(int i=0;i<25;i++) st[i]=0UL;\n"
"  for(int i=0;i<8;i++) {\n"
"    ulong v=0UL;\n"
"    for(int j=0;j<8;j++) v|=((ulong)in[i*8+j])<<(j*8);\n"
"    st[i]=v;\n"
"  }\n"
"  st[8] ^=0x0000000000000001UL;\n"
"  st[16]^=0x8000000000000000UL;\n"
"  keccak_f1600(st);\n"
"  for(int i=0;i<4;i++)\n"
"    for(int j=0;j<8;j++)\n"
"      hash[i*8+j]=(uchar)((st[i]>>(j*8))&0xFFu);\n"
"}\n"
"int hash_lt(const uchar *hash,__constant const uchar *diff) {\n"
"  for(int i=0;i<32;i++) {\n"
"    if(hash[i]<diff[i]) return 1;\n"
"    if(hash[i]>diff[i]) return 0;\n"
"  }\n"
"  return 0;\n"
"}\n"
"__kernel void mine(\n"
"  __constant uchar *challenge,\n"
"  __constant uchar *difficulty,\n"
"  __constant uchar *nonce_prefix,\n"
"  ulong base_counter,\n"
"  __global int  *found_flag,\n"
"  __global ulong *found_counter\n"
") {\n"
"  ulong tid=get_global_id(0);\n"
"  ulong my_counter=base_counter+tid;\n"
"  uchar input[64];\n"
"  for(int i=0;i<32;i++) input[i]=challenge[i];\n"
"  for(int i=0;i<24;i++) input[32+i]=nonce_prefix[i];\n"
"  input[56]=(uchar)(my_counter>>56);\n"
"  input[57]=(uchar)(my_counter>>48);\n"
"  input[58]=(uchar)(my_counter>>40);\n"
"  input[59]=(uchar)(my_counter>>32);\n"
"  input[60]=(uchar)(my_counter>>24);\n"
"  input[61]=(uchar)(my_counter>>16);\n"
"  input[62]=(uchar)(my_counter>> 8);\n"
"  input[63]=(uchar)(my_counter    );\n"
"  uchar hash[32];\n"
"  keccak256_64(input,hash);\n"
"  if(hash_lt(hash,difficulty)) {\n"
"    if(atomic_cmpxchg(found_flag,0,1)==0)\n"
"      found_counter[0]=my_counter;\n"
"  }\n"
"}\n";

/* ── fatal helpers ───────────────────────────────────────────────────────── */
static void fatal_json(const char *msg) {
    char esc[512];
    json_escape(esc, sizeof(esc), msg);
    printf("{\"type\":\"error\",\"message\":\"%s\"}\n", esc);
    fflush(stdout);
    exit(2);
}
static void fatal_cl(const char *what, cl_int err) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s failed: %d", what, (int)err);
    fatal_json(buf);
}

/* ═══════════════════════════════════════════════════════════════════════════
   main
   ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    /* defaults */
    const char *challenge_hex    = NULL;
    const char *difficulty_hex   = NULL;
    const char *nonce_prefix_hex = NULL;
    uint64_t    start_counter    = 0;
    size_t      batch_size       = 0;        /* 0 = auto */
    size_t      local_size       = 256;
    uint32_t    platform_index   = 0xFFFFFFFFu; /* 0xFFFF = auto-select GPU */
    uint32_t    device_index     = 0;
    uint64_t    progress_ms      = 2000;
    int         list_devices     = 0;

    for (int i = 1; i < argc; i++) {
#define NEXTARG (i + 1 < argc ? argv[++i] : (fprintf(stderr,"missing arg after %s\n",argv[i]),exit(2),(char*)0))
        if      (!strcmp(argv[i], "--challenge"))     challenge_hex    = NEXTARG;
        else if (!strcmp(argv[i], "--difficulty"))    difficulty_hex   = NEXTARG;
        else if (!strcmp(argv[i], "--nonce-prefix"))  nonce_prefix_hex = NEXTARG;
        else if (!strcmp(argv[i], "--start"))         start_counter    = parse_u64(NEXTARG);
        else if (!strcmp(argv[i], "--batch-size"))    batch_size       = (size_t)parse_u64(NEXTARG);
        else if (!strcmp(argv[i], "--local-size"))    local_size       = (size_t)parse_u64(NEXTARG);
        else if (!strcmp(argv[i], "--platform-index"))platform_index   = (uint32_t)parse_u64(NEXTARG);
        else if (!strcmp(argv[i], "--device-index"))  device_index     = (uint32_t)parse_u64(NEXTARG);
        else if (!strcmp(argv[i], "--progress-ms"))   progress_ms      = parse_u64(NEXTARG);
        else if (!strcmp(argv[i], "--list-devices"))  list_devices     = 1;
#undef NEXTARG
    }

    /* Load OpenCL */
    if (!load_opencl()) fatal_json("OpenCL runtime not found (OpenCL.dll / libOpenCL.so). Install your GPU driver or OpenCL runtime.");

    /* Enumerate platforms */
    cl_uint num_platforms = 0;
    if (cl.clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0)
        fatal_json("No OpenCL platforms found");
    cl_platform_id *platforms = (cl_platform_id *)calloc(num_platforms, sizeof(*platforms));
    cl.clGetPlatformIDs(num_platforms, platforms, NULL);

    /* --list-devices mode */
    if (list_devices) {
        printf("[");
        int first = 1;
        for (cl_uint pi = 0; pi < num_platforms; pi++) {
            char pname[256] = {0};
            get_str_platform(platforms[pi], CL_PLATFORM_NAME, pname, sizeof(pname));
            cl_uint nd = 0;
            cl.clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
            if (!nd) continue;
            cl_device_id *devs = (cl_device_id *)calloc(nd, sizeof(*devs));
            cl.clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, nd, devs, NULL);
            for (cl_uint di = 0; di < nd; di++) {
                char dname[256]={0}, dvendor[256]={0};
                cl_uint cu=0; size_t mwg=0;
                get_str_device(devs[di], CL_DEVICE_NAME,   dname,   sizeof(dname));
                get_str_device(devs[di], CL_DEVICE_VENDOR, dvendor, sizeof(dvendor));
                cl.clGetDeviceInfo(devs[di], CL_DEVICE_MAX_COMPUTE_UNITS,    sizeof(cu),  &cu,  NULL);
                cl.clGetDeviceInfo(devs[di], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mwg), &mwg, NULL);
                char dn[512]={0}, dv[512]={0}, pn[512]={0};
                json_escape(dn, sizeof(dn), dname);
                json_escape(dv, sizeof(dv), dvendor);
                json_escape(pn, sizeof(pn), pname);
                if (!first) printf(",");
                printf("{\"platform_index\":%u,\"device_index\":%u,\"platform\":\"%s\",\"device\":\"%s\",\"vendor\":\"%s\",\"cu\":%u,\"max_wg\":%zu}",
                    pi, di, pn, dn, dv, cu, mwg);
                first = 0;
            }
            free(devs);
        }
        printf("]\n");
        fflush(stdout);
        return 0;
    }

    /* Validate required args */
    if (!challenge_hex || !difficulty_hex || !nonce_prefix_hex) {
        fprintf(stderr, "usage: gpu_miner --challenge HEX64 --difficulty HEX64 --nonce-prefix HEX48 --start N\n");
        return 2;
    }
    uint8_t challenge[32], difficulty[32], nonce_prefix[24];
    if (parse_hex(challenge_hex,    challenge,    32) != 0) fatal_json("invalid --challenge hex");
    if (parse_hex(difficulty_hex,   difficulty,   32) != 0) fatal_json("invalid --difficulty hex");
    if (parse_hex(nonce_prefix_hex, nonce_prefix, 24) != 0) fatal_json("invalid --nonce-prefix hex");

    /* Select platform / device (auto = pick first platform with a GPU) */
    cl_platform_id chosen_platform = NULL;
    cl_device_id   chosen_device   = NULL;
    cl_uint        chosen_cu       = 0;
    size_t         chosen_mwg      = 0;
    char           chosen_pname[256] = {0};
    char           chosen_dname[256] = {0};
    char           chosen_dvendor[256] = {0};

    if (platform_index == 0xFFFFFFFFu) {
        /* Auto-select: iterate all platforms, find first GPU device */
        for (cl_uint pi = 0; pi < num_platforms && !chosen_device; pi++) {
            cl_uint nd = 0;
            if (cl.clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_GPU, 0, NULL, &nd) != CL_SUCCESS || nd == 0) continue;
            cl_device_id *devs = (cl_device_id *)calloc(nd, sizeof(*devs));
            cl.clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_GPU, nd, devs, NULL);
            /* Pick device with most CUs */
            for (cl_uint di = 0; di < nd; di++) {
                cl_uint cu = 0;
                cl.clGetDeviceInfo(devs[di], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
                if (cu > chosen_cu) {
                    chosen_cu = cu;
                    chosen_platform = platforms[pi];
                    chosen_device = devs[di];
                    cl.clGetDeviceInfo(devs[di], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(chosen_mwg), &chosen_mwg, NULL);
                    get_str_platform(platforms[pi], CL_PLATFORM_NAME, chosen_pname, sizeof(chosen_pname));
                    get_str_device(devs[di], CL_DEVICE_NAME,   chosen_dname,   sizeof(chosen_dname));
                    get_str_device(devs[di], CL_DEVICE_VENDOR, chosen_dvendor, sizeof(chosen_dvendor));
                }
            }
            free(devs);
        }
        if (!chosen_device) fatal_json("No OpenCL GPU device found. Install your GPU driver (NVIDIA, AMD, or Intel).");
    } else {
        if (platform_index >= num_platforms) fatal_json("--platform-index out of range");
        chosen_platform = platforms[platform_index];
        cl_uint nd = 0;
        if (cl.clGetDeviceIDs(chosen_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &nd) != CL_SUCCESS || nd == 0)
            fatal_json("No GPU devices on selected platform");
        cl_device_id *devs = (cl_device_id *)calloc(nd, sizeof(*devs));
        cl.clGetDeviceIDs(chosen_platform, CL_DEVICE_TYPE_GPU, nd, devs, NULL);
        if (device_index >= nd) { free(devs); fatal_json("--device-index out of range"); }
        chosen_device = devs[device_index];
        cl.clGetDeviceInfo(chosen_device, CL_DEVICE_MAX_COMPUTE_UNITS,    sizeof(chosen_cu),  &chosen_cu,  NULL);
        cl.clGetDeviceInfo(chosen_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(chosen_mwg), &chosen_mwg, NULL);
        get_str_platform(chosen_platform, CL_PLATFORM_NAME, chosen_pname, sizeof(chosen_pname));
        get_str_device(chosen_device, CL_DEVICE_NAME,   chosen_dname,   sizeof(chosen_dname));
        get_str_device(chosen_device, CL_DEVICE_VENDOR, chosen_dvendor, sizeof(chosen_dvendor));
        free(devs);
    }

    /* Auto batch size: target ~50 ms per dispatch = cu * mwg * 64 */
    if (batch_size == 0) {
        size_t auto_batch = (size_t)chosen_cu * chosen_mwg * 64;
        if (auto_batch < 262144)  auto_batch = 262144;
        if (auto_batch > 16777216) auto_batch = 16777216;
        batch_size = auto_batch;
    }
    /* Align to local_size */
    if (local_size > chosen_mwg && chosen_mwg > 0) local_size = chosen_mwg;
    if (batch_size % local_size != 0)
        batch_size = ((batch_size + local_size - 1) / local_size) * local_size;

    /* Report device */
    {
        char dn[512]={0}, dv[512]={0}, pn[512]={0};
        json_escape(dn, sizeof(dn), chosen_dname);
        json_escape(dv, sizeof(dv), chosen_dvendor);
        json_escape(pn, sizeof(pn), chosen_pname);
        printf("{\"type\":\"device\",\"platform\":\"%s\",\"device\":\"%s\",\"vendor\":\"%s\",\"cu\":%u,\"max_wg\":%zu,\"batch_size\":%zu,\"local_size\":%zu}\n",
            pn, dn, dv, chosen_cu, chosen_mwg, batch_size, local_size);
        fflush(stdout);
    }

    /* Build OpenCL program */
    cl_int err;
    cl_context ctx = cl.clCreateContext(NULL, 1, &chosen_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) fatal_cl("clCreateContext", err);
    cl_command_queue queue = cl.clCreateCommandQueue(ctx, chosen_device, 0, &err);
    if (err != CL_SUCCESS) fatal_cl("clCreateCommandQueue", err);

    cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, NULL, &err);
    if (err != CL_SUCCESS) fatal_cl("clCreateProgramWithSource", err);
    err = cl.clBuildProgram(prog, 1, &chosen_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        cl.clGetProgramBuildInfo(prog, chosen_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = (char *)calloc(log_sz + 1, 1);
        if (log) {
            cl.clGetProgramBuildInfo(prog, chosen_device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
            fprintf(stderr, "kernel build log:\n%s\n", log);
            free(log);
        }
        fatal_cl("clBuildProgram", err);
    }

    cl_kernel kernel = cl.clCreateKernel(prog, "mine", &err);
    if (err != CL_SUCCESS) fatal_cl("clCreateKernel", err);

    /* Buffers */
    cl_mem challenge_buf    = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, challenge,    &err); if (err) fatal_cl("clCreateBuffer(challenge)", err);
    cl_mem difficulty_buf   = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, difficulty,   &err); if (err) fatal_cl("clCreateBuffer(difficulty)", err);
    cl_mem nonce_prefix_buf = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 24, nonce_prefix, &err); if (err) fatal_cl("clCreateBuffer(prefix)", err);
    cl_mem found_flag_buf   = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_int),  NULL, &err); if (err) fatal_cl("clCreateBuffer(flag)", err);
    cl_mem found_ctr_buf    = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), NULL, &err); if (err) fatal_cl("clCreateBuffer(counter)", err);

    /* Set fixed kernel args (challenge/difficulty/prefix don't change per batch) */
    err  = cl.clSetKernelArg(kernel, 0, sizeof(challenge_buf),    &challenge_buf);
    err |= cl.clSetKernelArg(kernel, 1, sizeof(difficulty_buf),   &difficulty_buf);
    err |= cl.clSetKernelArg(kernel, 2, sizeof(nonce_prefix_buf), &nonce_prefix_buf);
    /* arg 3 = base_counter (set per batch) */
    err |= cl.clSetKernelArg(kernel, 4, sizeof(found_flag_buf),   &found_flag_buf);
    err |= cl.clSetKernelArg(kernel, 5, sizeof(found_ctr_buf),    &found_ctr_buf);
    if (err != CL_SUCCESS) fatal_cl("clSetKernelArg(fixed)", err);

    /* Mining loop */
    uint64_t next_counter = start_counter;
    uint64_t total_hashes = 0;
    uint64_t last_progress_t = now_ms();

    for (;;) {
        /* Reset output buffers */
        cl_int  flag  = 0;
        cl_ulong ctr  = 0;
        err  = cl.clEnqueueWriteBuffer(queue, found_flag_buf, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL);
        err |= cl.clEnqueueWriteBuffer(queue, found_ctr_buf,  CL_TRUE, 0, sizeof(ctr),  &ctr,  0, NULL, NULL);
        if (err != CL_SUCCESS) fatal_cl("clEnqueueWriteBuffer(reset)", err);

        /* Set per-batch counter */
        cl_ulong base = (cl_ulong)next_counter;
        err = cl.clSetKernelArg(kernel, 3, sizeof(base), &base);
        if (err != CL_SUCCESS) fatal_cl("clSetKernelArg(counter)", err);

        /* Dispatch */
        size_t gs = batch_size;
        size_t ls = local_size;
        err = cl.clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gs, &ls, 0, NULL, NULL);
        if (err != CL_SUCCESS) fatal_cl("clEnqueueNDRangeKernel", err);
        err = cl.clFinish(queue);
        if (err != CL_SUCCESS) fatal_cl("clFinish", err);

        total_hashes  += (uint64_t)batch_size;
        next_counter  += (uint64_t)batch_size;

        /* Read flag */
        err = cl.clEnqueueReadBuffer(queue, found_flag_buf, CL_TRUE, 0, sizeof(flag), &flag, 0, NULL, NULL);
        if (err != CL_SUCCESS) fatal_cl("clEnqueueReadBuffer(flag)", err);

        if (flag) {
            err = cl.clEnqueueReadBuffer(queue, found_ctr_buf, CL_TRUE, 0, sizeof(ctr), &ctr, 0, NULL, NULL);
            if (err != CL_SUCCESS) fatal_cl("clEnqueueReadBuffer(counter)", err);
            printf("{\"type\":\"found\",\"counter\":\"%" PRIu64 "\",\"hashes\":\"%" PRIu64 "\"}\n",
                (uint64_t)ctr, total_hashes);
            fflush(stdout);
            break;
        }

        /* Progress */
        uint64_t now = now_ms();
        if (now - last_progress_t >= progress_ms) {
            printf("{\"type\":\"progress\",\"hashes\":\"%" PRIu64 "\",\"nonce\":\"%" PRIu64 "\"}\n",
                total_hashes, next_counter);
            fflush(stdout);
            last_progress_t = now;
        }
    }

    /* Cleanup */
    cl.clReleaseMemObject(found_ctr_buf);
    cl.clReleaseMemObject(found_flag_buf);
    cl.clReleaseMemObject(nonce_prefix_buf);
    cl.clReleaseMemObject(difficulty_buf);
    cl.clReleaseMemObject(challenge_buf);
    cl.clReleaseKernel(kernel);
    cl.clReleaseProgram(prog);
    cl.clReleaseCommandQueue(queue);
    cl.clReleaseContext(ctx);
    free(platforms);
    return 0;
}

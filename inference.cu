#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file,
           const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << path << std::endl;
        return {};
    }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char *)&magic_number, 4);
    magic_number = __builtin_bswap32(magic_number);
    file.read((char *)&num_images, 4);
    num_images = __builtin_bswap32(num_images);
    file.read((char *)&num_rows, 4);
    num_rows = __builtin_bswap32(num_rows);
    file.read((char *)&num_cols, 4);
    num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(
        num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char *)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
        images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) /
                        0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << path << std::endl;
        return {};
    }
    int magic_number = 0, num_items = 0;
    file.read((char *)&magic_number, 4);
    magic_number = __builtin_bswap32(magic_number);
    file.read((char *)&num_items, 4);
    num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char *)buffer.data(), num_items);
    for (int i = 0; i < num_items; ++i) {
        labels[i] = static_cast<int>(buffer[i]);
    }
    return labels;
}

std::vector<float> read_param(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Cannot open parameter file: " << path << std::endl;
        return {};
    }
    std::vector<float> params;
    float param;
    while (file >> param) {
        params.push_back(param);
    }
    return params;
}

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define OFFSET2D(D0, x1, x0) ((x1) * (D0) + (x0))
#define OFFSET3D(D1, D0, x2, x1, x0) ((x2) * (D1) * (D0) + (x1) * (D0) + (x0))
#define OFFSET4D(D2, D1, D0, x3, x2, x1, x0)                                   \
    ((x3) * (D2) * (D1) * (D0) + (x2) * (D1) * (D0) + (x1) * (D0) + (x0))


// layer forward declarations

// conv2d_forward(input, weight, bias, output, Cin, Cout, H, W, kernel_size,
// padding)
void conv2d_forward(const float *input, const float *weight, const float *bias,
                    float *output, int batch, int Cin, int Cout, int H, int W,
                    int kernel_size, int padding);

// maxpool2d_forward(input, output, C, H, W)
void maxpool2d_forward(const float *input, float *output, int batch, int C,
                      int H, int W);

void linear_forward(const float *input, const float *weight, const float *bias,
                    float *output, int batch, int in_dim, int out_dim,
                    bool accumulate = false);

// IFNode_forward(V, I, S, N, threshold)
void IFNode_forward(float *V, const float *I, float *S, int batch, int N,
                    float threshold);

// Fused Conv2D + IFNode forward
void fused_conv2d_ifnode_forward(const float *input, const float *weight,
                                const float *bias, float *V, float *S,
                                int batch, int Cin, int Cout, int H, int W,
                                int kernel_size, int padding, float threshold);

// Fused GEMM + IFNode forward
void fused_linear_ifnode_forward(const float *input, const float *weight,
                                const float *bias, float *V, float *S,
                                int batch, int in_dim, int out_dim,
                                float threshold);

std::vector<int>
scnn_inference(const std::vector<std::vector<float>> &images,
              // Device pointers for parameters
              float *d_conv1_w, float *d_conv1_b, float *d_conv2_w,
              float *d_conv2_b, float *d_fc1_w, float *d_fc1_b, float *d_fc2_w,
              float *d_fc2_b, float *d_fc3_w, float *d_fc3_b,
              int batch_size = 8 // 新增：一次处理的样本数（默认8）
) {
    std::vector<int> predictions;
    const int num_images = images.size();
    predictions.reserve(num_images);

    // SNN-specific parameter, must match training
    const int T = 2;              // 时间步
    const float threshold = 0.20f; // 膜电位阈值

    const int Cin1 = 1;
    const int Cout1 = 8;
    const int H1 = 28;
    const int W1 = 28;
    const int ksize_conv1 = 3;
    const int padding_conv1 = 1;
    const int Cout2 = 16;
    const int Hout1 = (H1 - ksize_conv1 + 2 * padding_conv1) / 1 + 1; // 28
    const int Wout1 = (W1 - ksize_conv1 + 2 * padding_conv1) / 1 + 1; // 28
    const int H2 = Hout1 / 2;                                         // 14
    const int W2 = Wout1 / 2;                                         // 14
    const int ksize_conv2 = 3;
    const int padding_conv2 = 1;
    const int Hout2 = (H2 - ksize_conv2 + 2 * padding_conv2) / 1 + 1; // 14
    const int Wout2 = (W2 - ksize_conv2 + 2 * padding_conv2) / 1 + 1; // 14
    const int H3 = Hout2 / 2;                                         // 7
    const int W3 = Wout2 / 2;                                         // 7
    const int fc1_in_dim = Cout2 * H3 * W3; // 16 * 7 * 7 = 784
    const int fc1_out_dim = 96;
    const int fc2_out_dim = 64;
    const int fc3_out_dim = 10;

    // cudaStream_t stream;
    // cudaStreamCreate(&stream);

    ////////////////////////////////////////
    // 分配中间 GPU 缓冲区

    // input: [batch, Cin1, H1, W1]
    float *d_input = nullptr;
    checkCudaErrors(cudaMalloc(&d_input, batch_size * Cin1 * H1 * W1 * sizeof(float)));

    // Conv1 输出: [batch, Cout1, Hout1, Wout1]
    float *d_conv1_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_conv1_out, batch_size * Cout1 * Hout1 * Wout1 * sizeof(float)));

    // Pool1 输出: [batch, Cout1, H2, W2]
    float *d_pool1_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_pool1_out, batch_size * Cout1 * H2 * W2 * sizeof(float)));

    // Conv2 输出: [batch, Cout2, Hout2, Wout2]
    float *d_conv2_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_conv2_out, batch_size * Cout2 * Hout2 * Wout2 * sizeof(float)));

    // Pool2 输出: [batch, Cout2, H3, W3]
    float *d_pool2_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_pool2_out, batch_size * Cout2 * H3 * W3 * sizeof(float)));

    // Flatten 后 FC1 输入: [batch, fc1_in_dim]
    float *d_fc1_in = d_pool2_out; // 复用池化输出缓冲(展平层不改变数据布局)
    float *d_fc1_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_fc1_out, batch_size * fc1_out_dim * sizeof(float)));

    // FC2 输出: [batch, fc2_out_dim]
    float *d_fc2_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_fc2_out, batch_size * fc2_out_dim * sizeof(float)));

    // FC3 输出: [batch, fc3_out_dim]
    float *d_fc3_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_fc3_out, batch_size * fc3_out_dim * sizeof(float)));

    // 分配膜电位缓冲
    float *d_V1 = nullptr, *d_V2 = nullptr, *d_V3 = nullptr, *d_V4 = nullptr;
    checkCudaErrors(cudaMalloc(&d_V1, batch_size * Cout1 * Hout1 * Wout1 * sizeof(float))); // Conv1
    checkCudaErrors(cudaMalloc(&d_V2, batch_size * Cout2 * Hout2 * Wout2 * sizeof(float))); // Conv2
    checkCudaErrors(cudaMalloc(&d_V3, batch_size * fc1_out_dim * sizeof(float))); // FC1
    checkCudaErrors(cudaMalloc(&d_V4, batch_size * fc2_out_dim * sizeof(float))); // FC2

    // 输出脉冲累加缓冲: [batch, fc3_out_dim]
    float *d_output_spikes = d_fc3_out; // 复用 FC3 输出缓冲来累加！

    // --- Loop over each image ---
    for (int i = 0; i < num_images; i += batch_size) {
        int B = std::min(batch_size, num_images - i); // 当前批大小（最后一批可能小）
        // Host => Device 拷贝B张图片
        // pack B images into a contiguous host buffer then memcpy
        std::vector<float> h_batch_in(B * Cin1 * H1 * W1);
        for (int b = 0; b < B; ++b) {
            const auto &img = images[i + b];
            memcpy(h_batch_in.data() + b * Cin1 * H1 * W1, img.data(),
                    Cin1 * H1 * W1 * sizeof(float));
        }
        checkCudaErrors(cudaMemcpy(d_input, h_batch_in.data(),
                                h_batch_in.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

        // 清空膜电位
        checkCudaErrors(cudaMemset(d_V1, 0, batch_size * Cout1 * Hout1 * Wout1 * sizeof(float)));
        checkCudaErrors(cudaMemset(d_V2, 0, batch_size * Cout2 * Hout2 * Wout2 * sizeof(float)));
        checkCudaErrors(cudaMemset(d_V3, 0, batch_size * fc1_out_dim * sizeof(float)));
        checkCudaErrors(cudaMemset(d_V4, 0, batch_size * fc2_out_dim * sizeof(float)));
        // 清空输出脉冲缓冲
        checkCudaErrors(cudaMemset(d_output_spikes, 0, batch_size * fc3_out_dim * sizeof(float)));


        // 时间步推理循环
        for (int t = 0; t < T; ++t) {
        // --- Conv1 ---
        // conv2d_forward(d_input, d_conv1_w, d_conv1_b, d_conv1_out, B,
        //                Cin1, Cout1, H1, W1, ksize_conv1, padding_conv1);
        // IFNode_forward(d_V1, d_conv1_out, d_conv1_out, B, Cout1 * Hout1 *
        // Wout1, threshold);
        fused_conv2d_ifnode_forward(d_input, d_conv1_w, d_conv1_b, d_V1,
                                    d_conv1_out, B, Cin1, Cout1, H1, W1,
                                    ksize_conv1, padding_conv1, threshold);
        maxpool2d_forward(d_conv1_out, d_pool1_out, B, Cout1, Hout1, Wout1);

        // --- Conv2 ---
        // conv2d_forward(d_pool1_out, d_conv2_w, d_conv2_b, d_conv2_out, B,
        //                Cout1, Cout2, H2, W2, ksize_conv2, padding_conv2);
        // IFNode_forward(d_V2, d_conv2_out, d_conv2_out, B, Cout2 * Hout2 *
        // Wout2, threshold);
        fused_conv2d_ifnode_forward(d_pool1_out, d_conv2_w, d_conv2_b, d_V2,
                                    d_conv2_out, B, Cout1, Cout2, H2, W2,
                                    ksize_conv2, padding_conv2, threshold);
        maxpool2d_forward(d_conv2_out, d_pool2_out, B, Cout2, Hout2, Wout2);

        // --- Flatten ---
        // flatten_forward(d_pool2_out, d_fc1_in, Cout2, H3, W3);

        // --- FC1 ---
        // linear_forward(d_fc1_in, d_fc1_w, d_fc1_b, d_fc1_out, B, fc1_in_dim,
        // fc1_out_dim); IFNode_forward(d_V3, d_fc1_out, d_fc1_out, B,
        // fc1_out_dim, threshold);
        fused_linear_ifnode_forward(d_fc1_in, d_fc1_w, d_fc1_b, d_V3, d_fc1_out,
                                    B, fc1_in_dim, fc1_out_dim, threshold);

        // --- FC2 ---
        // linear_forward(d_fc1_out, d_fc2_w, d_fc2_b, d_fc2_out, B, fc1_out_dim,
        // fc2_out_dim); IFNode_forward(d_V4, d_fc2_out, d_fc2_out, B,
        // fc2_out_dim, threshold);
        fused_linear_ifnode_forward(d_fc1_out, d_fc2_w, d_fc2_b, d_V4, d_fc2_out,
                                    B, fc1_out_dim, fc2_out_dim, threshold);

        // --- FC3 (输出层) ---
        linear_forward(d_fc2_out, d_fc3_w, d_fc3_b, d_fc3_out, B, fc2_out_dim,
                        fc3_out_dim, true); // 累加到输出缓冲
        }

        // 将累积的输出脉冲传回 CPU
        // Host-side output buffer
        std::vector<float> h_out(batch_size * fc3_out_dim);
        checkCudaErrors(cudaMemcpy(h_out.data(), d_output_spikes,
                                B * fc3_out_dim * sizeof(float),
                                cudaMemcpyDeviceToHost));

        // for each sample in batch, pick argmax and push to predictions
        for (int b = 0; b < B; ++b) {
            float *row = h_out.data() + b * fc3_out_dim;
            int pred = std::distance(row, std::max_element(row, row + fc3_out_dim));
            predictions.push_back(pred);
        }
    }

    // Free gpu buffers
    cudaFree(d_input);
    cudaFree(d_conv1_out);
    cudaFree(d_pool1_out);
    cudaFree(d_conv2_out);
    cudaFree(d_pool2_out);
    cudaFree(d_fc1_in);
    cudaFree(d_fc1_out);
    cudaFree(d_fc2_out);
    cudaFree(d_fc3_out);
    cudaFree(d_V1);
    cudaFree(d_V2);
    cudaFree(d_V3);
    cudaFree(d_V4);
    cudaFree(d_output_spikes);

    // cudaStreamSynchronize(stream);
    // cudaStreamDestroy(stream);

    return predictions;
}


////////////////////////////////////////////////////////////////////////
// [Inline PTX Optimization] Vectorized Load (Global -> Register -> Shared)
__device__ __forceinline__ void copy_float4_ptx(const float* src_global, float* dst_shared) {
    float r0, r1, r2, r3;
    // Vector Load from Global Memory
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "l"(src_global)
    );
    // Vector Store to Shared Memory
    // 这里需要注意：Shared Memory 指针在 PTX 中通常是 32-bit (或者 .u64 generic)
    // 为了安全，我们用 __cvta_generic_to_shared 将指针转为 SMEM 空间偏移量
    unsigned int smem_ptr_uint;
    asm volatile(
        "{ .reg .u64 %addr; cvta.to.shared.u64 %addr, %1; cvt.u32.u64 %0, %addr; }"
        : "=r"(smem_ptr_uint)
        : "l"(dst_shared)
    );
    asm volatile(
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "r"(smem_ptr_uint), "f"(r0), "f"(r1), "f"(r2), "f"(r3)
    );
}


////////////////////////////////////////////////////////////////////////
// GEMM kernel with tiling and shared memory
#define TILE_M 32
#define TILE_N 32
#define TILE_K 64
#define RM 4
#define RN 2

__global__ void
tiled_gemm(const float *__restrict__ A, // M x K
           const float *__restrict__ B, // N x K  (weight rows = output neurons)
           const float *__restrict__ bias, // N
           float *__restrict__ C,          // M x N
           int M, int K, int N, bool accumulate)
{
    // block tile origin
    const int block_row = blockIdx.y; // tile index along M (batch)
    const int block_col = blockIdx.x; // tile index along N (out_dim)

    // thread local coordinates inside tile
    const int local_row = threadIdx.y * RM; // 0..TILE_M-1
    const int local_col = threadIdx.x * RN; // 0..TILE_N-1

    // global coordinates (row, col) this thread will compute
    const int row = block_row * TILE_M + local_row;
    const int col = block_col * TILE_N + local_col;

    // Shared memory for a tile of A (TILE_M x TILE_K) and a tile of B (TILE_N x TILE_K). 
    // We declare with fixed sizes -- safe because TILE_* are compile-time constants.
    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_N][TILE_K];

    // Accumulator
    float acc[RM][RN];
    #pragma unroll
    for (int r = 0; r < RM; ++r)
        #pragma unroll
        for (int c = 0; c < RN; ++c)
            acc[r][c] = 0.0f;

    // Loop over K dimension by tiles
    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        const int thread_linear = OFFSET2D(blockDim.x, threadIdx.y, threadIdx.x); // Linear index among threads inside block
        const int block_thread_count = blockDim.x * blockDim.y;

        // ---- Cooperative load of A_tile (size TILE_M x TILE_K) ----
        // We let all threads in block cooperatively load the entire A_tile.
        const int A_tile_elems = TILE_M * TILE_K; // total elements to load for A_tile:
        for (int idx = thread_linear * 4; idx < A_tile_elems; idx += block_thread_count * 4) {
            int i = idx / TILE_K; // 0..TILE_M-1
            int k = idx % TILE_K; // 0..TILE_K-1
            int global_r = block_row * TILE_M + i;
            int global_k = k0 + k;
            if (global_r < M && global_k + 3 < K) {
                copy_float4_ptx(&A[OFFSET2D(K, global_r, global_k)], &sA[i][k]);
            } else {
                for (int v=0; v<4; ++v)
                    sA[i][k+v] = 0.0f;
            }
        }

        // ---- Cooperative load of B_tile (size TILE_K x TILE_N) ----
        // Want sB[k][j] = B[(col_base + j) * K + (k0 + k)]
        const int B_tile_elems = TILE_K * TILE_N;
        for (int idx = thread_linear * 4; idx < B_tile_elems; idx += block_thread_count * 4) {
            int j = idx / TILE_K; // 0..TILE_N-1
            int k = idx % TILE_K; // 0..TILE_K-1
            int global_k = k0 + k;
            int global_c = block_col * TILE_N + j;
            if (global_c < N && global_k + 3 < K) {
                copy_float4_ptx(&B[OFFSET2D(K, global_c, global_k)], &sB[j][k]);
            } else {
                for (int v=0; v<4; ++v)
                    sB[j][k+v] = 0.0f;
            }
        }

        // Ensure tile loads complete
        __syncthreads();

        // ---- compute local multiply-accumulate for this k-tile ----
        // local_row in 0..TILE_M-1, local_col in 0..TILE_N-1
        for (int kk = 0; kk < TILE_K; ++kk) {
        #pragma unroll
        for (int r = 0; r < RM; ++r)
            #pragma unroll
            for (int c = 0; c < RN; ++c)
                acc[r][c] += sA[local_row + r][kk] * sB[local_col + c][kk];
        }

        __syncthreads(); // ensure no thread reads sA/sW while they're being
                        // overwritten next iteration
    } // end for k0

    // After accumulation, add bias and write to C
    #pragma unroll
    for (int r = 0; r < RM && row + r < M; ++r) {
        #pragma unroll
        for (int c = 0; c < RN && col + c < N; ++c) {
            float out_val = acc[r][c] + bias[col + c];
            if (accumulate) {
                C[OFFSET2D(N, row + r, col + c)] += out_val;
            } else {
                C[OFFSET2D(N, row + r, col + c)] = out_val;
            }
        }
    }
}
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Conv2D kernel with tiling and shared memory
#define TILE_H 14
#define TILE_W 14
#define OUT_PER_BLOCK 4
#define KERNEL_SIZE 3

// blockDim: (TILE_W, TILE_H)
// gridDim.x = CEIL_DIV(Wout, TILE_W)
// gridDim.y = CEIL_DIV(Hout, TILE_H)
// gridDim.z = batch * CEIL(Cout, OUT_PER_BLOCK)
__global__ void
tiled_conv2d_kernel(const float *__restrict__ input,  // [batch, Cin, H, W]
                    const float *__restrict__ weight, // [Cout, Cin, K, K]
                    const float *__restrict__ bias,   // [Cout]
                    float *__restrict__ output, // [batch, Cout, Hout, Wout]
                    int Cin, int Cout, int H, int W, int padding, int Hout,
                    int Wout)
{
    // Pad Shared Memory 避免 Bank Conflict
    __shared__ float tile_in[TILE_H + KERNEL_SIZE - 1][TILE_W + KERNEL_SIZE + 1]; // 共享输入tile
    __shared__ alignas(16) float tile_weight[KERNEL_SIZE][KERNEL_SIZE][OUT_PER_BLOCK]; // 共享权重tile

    int n = blockIdx.z / CEIL(Cout, OUT_PER_BLOCK);
    int cout_group_id = blockIdx.z % CEIL(Cout, OUT_PER_BLOCK);
    int out_h = blockIdx.y * TILE_H + threadIdx.y;
    int out_w = blockIdx.x * TILE_W + threadIdx.x;

    int in_base = n * Cin * H * W;
    int out_base = n * Cout * Hout * Wout;
    int out_ch_base = cout_group_id * OUT_PER_BLOCK;

    float acc[OUT_PER_BLOCK];
    #pragma unroll
    for (int oc = 0; oc < OUT_PER_BLOCK; ++oc) {
        int oc_global = out_ch_base + oc;
        acc[oc] = oc_global < Cout ? bias[oc_global] : 0.0f;
    }

    int ih0 = blockIdx.y * TILE_H - padding;
    int iw0 = blockIdx.x * TILE_W - padding;

    for (int cin = 0; cin < Cin; ++cin) {
        const float *input_cin = input + in_base + cin * H * W;

        // Cooperative load of this cin's tile
        const int tile_in_elems = sizeof(tile_in) / sizeof(float);
        const int thread_linear = OFFSET2D(blockDim.x, threadIdx.y, threadIdx.x); // Linear index among threads inside block
        const int block_thread_count = blockDim.x * blockDim.y;
        for (int idx = thread_linear * 4; idx < tile_in_elems; idx += block_thread_count * 4) {
            int i = idx / (TILE_W + KERNEL_SIZE + 1);
            int j = idx % (TILE_W + KERNEL_SIZE + 1);
            int ih = ih0 + i;
            int iw = iw0 + j;
            if (ih >= 0 && ih < H && iw >= 0 && iw + 3 < W) {
                copy_float4_ptx(&input_cin[OFFSET2D(W, ih, iw)], &tile_in[i][j]);
            } else {
                for (int v=0; v<4; ++v) {
                    float val = 0.0f;
                    if (ih >= 0 && ih < H && iw + v >= 0 && iw + v < W) {
                        val = input_cin[OFFSET2D(W, ih, iw + v)];
                    }
                    tile_in[i][j + v] = val;
                }
            }
        }

        // Cooperative load of this cin's weight tile
        // [Cout, Cin, K, K] => [Cin, K, K, Cout]
        const int tile_weight_elems = sizeof(tile_weight) / sizeof(float);
        for (int idx = thread_linear; idx < tile_weight_elems; idx += block_thread_count) {
            int oc_local = idx / (KERNEL_SIZE * KERNEL_SIZE);
            int i = (idx / KERNEL_SIZE) % KERNEL_SIZE;
            int j = idx % KERNEL_SIZE;
            int oc_global = out_ch_base + oc_local;
            if (oc_global >= Cout) continue;
            tile_weight[i][j][oc_local] = weight[OFFSET4D(Cin, KERNEL_SIZE, KERNEL_SIZE, oc_global, cin, i, j)];
        }

        __syncthreads();

        // 每个线程计算一个输出像素 (out_h, out_w)
        if (out_h < Hout && out_w < Wout) {
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    float in_val = tile_in[threadIdx.y + kh][threadIdx.x + kw];
                    float w0, w1, w2, w3;
                    float* w_ptr = tile_weight[kh][kw];
                    // 向量读取 float4: Generic Pointer (64bit) -> Shared Offset (32bit) -> Load
                    unsigned int smem_addr;
                    asm volatile(
                        "{ .reg .u64 %addr; cvta.to.shared.u64 %addr, %5; cvt.u32.u64 %0, %addr; }\n"
                        "ld.shared.v4.f32 {%1, %2, %3, %4}, [%0];"
                        : "=r"(smem_addr), "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                        : "l"(w_ptr)
                    );

                    // acc[0] += in_val * w0
                    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[0]) : "f"(in_val), "f"(w0));
                    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[1]) : "f"(in_val), "f"(w1));
                    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[2]) : "f"(in_val), "f"(w2));
                    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[3]) : "f"(in_val), "f"(w3));
                }
            }
        }

        __syncthreads();
    } // end for cin

    if (out_h < Hout && out_w < Wout) {
        #pragma unroll
        for (int oc_local = 0; oc_local < OUT_PER_BLOCK; ++oc_local) {
            int oc_global = out_ch_base + oc_local;
            if (oc_global >= Cout) continue;
            output[out_base + OFFSET3D(Hout, Wout, oc_global, out_h, out_w)] = acc[oc_local];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// fuesd Conv2D + IFNode kernel with tiling and shared memory
__global__ void fused_conv2d_ifnode(const float *__restrict__ input,
                                    const float *__restrict__ weight,
                                    const float *__restrict__ bias,
                                    float *__restrict__ V, // 电位状态
                                    float *__restrict__ S, // 输出脉冲
                                    int Cin, int Cout, int H, int W,
                                    int padding, int Hout, int Wout,
                                    float threshold)
{
    // Pad Shared Memory 避免 Bank Conflict
    __shared__ float tile_in[TILE_H + KERNEL_SIZE - 1][TILE_W + KERNEL_SIZE - 1]; // 共享输入tile
    __shared__ alignas(16) float tile_weight[KERNEL_SIZE][KERNEL_SIZE][OUT_PER_BLOCK]; // 共享权重tile

    int n = blockIdx.z / CEIL(Cout, OUT_PER_BLOCK);
    int cout_group_id = blockIdx.z % CEIL(Cout, OUT_PER_BLOCK);
    int out_h = blockIdx.y * TILE_H + threadIdx.y;
    int out_w = blockIdx.x * TILE_W + threadIdx.x;

    int in_base = n * Cin * H * W;
    int out_base = n * Cout * Hout * Wout;
    int out_ch_base = cout_group_id * OUT_PER_BLOCK;

    float acc[OUT_PER_BLOCK];
    #pragma unroll
    for (int oc = 0; oc < OUT_PER_BLOCK; ++oc) {
        int oc_global = out_ch_base + oc;
        acc[oc] = oc_global < Cout ? bias[oc_global] : 0.0f;
    }

    int ih0 = blockIdx.y * TILE_H - padding;
    int iw0 = blockIdx.x * TILE_W - padding;

    for (int cin = 0; cin < Cin; ++cin) {
        const float *input_cin = input + in_base + cin * H * W;

        // Cooperative load of this cin's tile
        const int tile_in_elems = sizeof(tile_in) / sizeof(float);

        const int thread_linear = OFFSET2D(blockDim.x, threadIdx.y, threadIdx.x); // Linear index among threads inside block
        const int block_thread_cnt = blockDim.x * blockDim.y;
        for (int idx = thread_linear; idx < tile_in_elems; idx += block_thread_cnt) {
            int i = idx / (TILE_W + KERNEL_SIZE - 1);
            int j = idx % (TILE_W + KERNEL_SIZE - 1);
            int ih = ih0 + i;
            int iw = iw0 + j;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                tile_in[i][j] = input_cin[OFFSET2D(W, ih, iw)];
            } else {
                tile_in[i][j] = 0.0f;
            }
        }

        // Cooperative load of this cin's weight tile
        // [Cout, Cin, K, K] => [Cin, K, K, Cout]
        const int tile_weight_elems = sizeof(tile_weight) / sizeof(float);
        for (int idx = thread_linear; idx < tile_weight_elems; idx += block_thread_cnt) {
            int oc_local = idx / (KERNEL_SIZE * KERNEL_SIZE);
            int i = (idx / KERNEL_SIZE) % KERNEL_SIZE;
            int j = idx % KERNEL_SIZE;
            int oc_global = out_ch_base + oc_local;
            if (oc_global >= Cout) continue;
            tile_weight[i][j][oc_local] = weight[OFFSET4D(Cin, KERNEL_SIZE, KERNEL_SIZE, oc_global, cin, i, j)];
        }

        __syncthreads();

        // 每个线程计算一个输出像素 (out_h, out_w)
        if (out_h < Hout && out_w < Wout) {
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    float in_val = tile_in[threadIdx.y + kh][threadIdx.x + kw];
                    float w0, w1, w2, w3;
                    float* w_ptr = tile_weight[kh][kw];
                    // 向量读取 float4: Generic Pointer (64bit) -> Shared Offset (32bit) -> Load
                    unsigned int smem_addr;
                    asm volatile(
                        "{ .reg .u64 %addr; cvta.to.shared.u64 %addr, %5; cvt.u32.u64 %0, %addr; }\n"
                        "ld.shared.v4.f32 {%1, %2, %3, %4}, [%0];"
                        : "=r"(smem_addr), "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                        : "l"(w_ptr)
                    );

                    // acc[0] += in_val * w0
                    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[0]) : "f"(in_val), "f"(w0));
                    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[1]) : "f"(in_val), "f"(w1));
                    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[2]) : "f"(in_val), "f"(w2));
                    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[3]) : "f"(in_val), "f"(w3));
                }
            }
        }

        __syncthreads();
    } // end for cin

    // 融合 bias + IFNode 更新
    if (out_h < Hout && out_w < Wout) {
        #pragma unroll
        for (int oc = 0; oc < OUT_PER_BLOCK && out_ch_base + oc < Cout; ++oc) {
            int out_idx = out_base + OFFSET3D(Hout, Wout, out_ch_base + oc, out_h, out_w);
            float v = V[out_idx] + acc[oc];
            if (v >= threshold) {
                S[out_idx] = 1.0f;
                v = 0.0f;
            } else {
                S[out_idx] = 0.0f;
            }
            V[out_idx] = v;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Fused GEMM + IFNode forward

__global__ void fused_gemm_ifnode_kernel(const float *__restrict__ A, // [M, K]
                                        const float *__restrict__ B, // [N, K]
                                        const float *__restrict__ bias, // [N]
                                        float *__restrict__ V, // 电位状态
                                        float *__restrict__ S, // 输出脉冲
                                        int M, int K, int N, float threshold) {
    // block tile origin
    const int block_row = blockIdx.y; // tile index along M (batch)
    const int block_col = blockIdx.x; // tile index along N (out_dim)

    // thread local coordinates inside tile
    const int local_row = threadIdx.y * RM; // 0..TILE_M-1
    const int local_col = threadIdx.x * RN; // 0..TILE_N-1

    // global coordinates (row, col) this thread will compute
    const int row = block_row * TILE_M + local_row;
    const int col = block_col * TILE_N + local_col;

    // Shared memory for a tile of A (TILE_M x TILE_K) and a tile of B (TILE_N x
    // TILE_K). We declare with fixed sizes -- safe because TILE_* are
    // compile-time constants.
    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_N][TILE_K];

    // Accumulator
    float acc[RM][RN];
    #pragma unroll
    for (int r = 0; r < RM; ++r)
        #pragma unroll
        for (int c = 0; c < RN; ++c)
            acc[r][c] = 0.0f;

    // Loop over K dimension by tiles
    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        const int thread_linear = OFFSET2D(blockDim.x, threadIdx.y, threadIdx.x); // Linear index among threads inside block
        const int block_thread_count = blockDim.x * blockDim.y;

        // ---- Cooperative load of A_tile (size TILE_M x TILE_K) ----
        // We let all threads in block cooperatively load the entire A_tile.
        const int A_tile_elems = TILE_M * TILE_K; // total elements to load for A_tile:
        for (int idx = thread_linear * 4; idx < A_tile_elems; idx += block_thread_count * 4) {
            int i = idx / TILE_K; // 0..TILE_M-1
            int k = idx % TILE_K; // 0..TILE_K-1
            int global_r = block_row * TILE_M + i;
            int global_k = k0 + k;
            if (global_r < M && global_k + 3 < K) {
                copy_float4_ptx(&A[OFFSET2D(K, global_r, global_k)], &sA[i][k]);
            } else {
                for (int v=0; v<4; ++v)
                sA[i][k+v] = 0.0f;
            }
        }

        // ---- Cooperative load of B_tile (size TILE_K x TILE_N) ----
        // Want sB[k][j] = B[(col_base + j) * K + (k0 + k)]
        const int B_tile_elems = TILE_K * TILE_N;
        for (int idx = thread_linear * 4; idx < B_tile_elems; idx += block_thread_count * 4) {
            int j = idx / TILE_K; // 0..TILE_N-1
            int k = idx % TILE_K; // 0..TILE_K-1
            int global_k = k0 + k;
            int global_c = block_col * TILE_N + j;
            if (global_c < N && global_k + 3 < K) {
                copy_float4_ptx(&B[OFFSET2D(K, global_c, global_k)], &sB[j][k]);
            } else {
                for (int v=0; v<4; ++v)
                    sB[j][k+v] = 0.0f;
            }
        }

        // Ensure tile loads complete
        __syncthreads();

        // ---- compute local multiply-accumulate for this k-tile ----
        // local_row in 0..TILE_M-1, local_col in 0..TILE_N-1
        for (int kk = 0; kk < TILE_K; ++kk) {
        #pragma unroll
        for (int r = 0; r < RM; ++r)
            #pragma unroll
            for (int c = 0; c < RN; ++c)
                acc[r][c] += sA[local_row + r][kk] * sB[local_col + c][kk];
        }

        __syncthreads(); // ensure no thread reads sA/sW while they're being overwritten next iteration
    } // end for k0

    // 融合 bias + IFNode 更新
    #pragma unroll
    for (int r = 0; r < RM && row + r < M; ++r) {
        #pragma unroll
        for (int c = 0; c < RN && col + c < N; ++c) {
            const int out_idx = OFFSET2D(N, row + r, col + c);
            float out_val = acc[r][c] + bias[col + c];
            float v = V[out_idx] + out_val;
            if (v >= threshold) {
                S[out_idx] = 1.0f;
                v = 0.0f;
            } else {
                S[out_idx] = 0.0f;
            }
            V[out_idx] = v;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////

// 卷积层实现
// input:  [batch, Cin, H, W]
// weight: [Cout, Cin, K, K]
// bias:   [Cout]
// output: [batch, Cout, Hout, Wout]
// We will encode batch and Cout into grid.z: grid.z = batch * Cout
void conv2d_forward(const float *input, const float *weight, const float *bias,
                    float *output, int batch, int Cin, int Cout, int H, int W,
                    int kernel_size, int padding) {
    int Hout = (H - kernel_size + 2 * padding) / 1 + 1;
    int Wout = (W - kernel_size + 2 * padding) / 1 + 1;

    dim3 block(TILE_W, TILE_H);
    dim3 grid(CEIL(Wout, TILE_W), CEIL(Hout, TILE_H),
                batch * CEIL(Cout, OUT_PER_BLOCK));
    assert(kernel_size == 3); // 3x3卷积核
    tiled_conv2d_kernel<<<grid, block>>>(input, weight, bias, output, Cin,
                                         Cout, H, W, padding, Hout, Wout);
}


__global__ void maxpool2d_kernel(const float *__restrict__ input,
                                float *__restrict__ output, int C, int H,
                                int W) {
    int c = blockIdx.z % C;
    int n = blockIdx.z / C;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H / 2, W_out = W / 2;
    if (h_out >= H_out || w_out >= W_out)
        return;

    float max_val = 0.0f;
    int in_base = n * (C * H * W) + c * (H * W);
    int out_base = n * (C * H_out * W_out) + c * (H_out * W_out);

    #pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < 2; ++kw) {
            int ih = h_out * 2 + kh;
            int iw = w_out * 2 + kw;
            float val = input[in_base + OFFSET2D(W, ih, iw)];
            max_val = fmaxf(max_val, val);
        }
    }

    output[out_base + OFFSET2D(W_out, h_out, w_out)] = max_val;
}

// 最大池化层实现
// input:  [batch, C, H, W]
// output: [batch, C, H/2, W/2]
// We'll encode batch and channel into grid.z: grid.z = batch * C
void maxpool2d_forward(const float *input, float *output, int batch, int C,
                      int H, int W) {
    dim3 block(16, 16);
    dim3 grid(CEIL(W / 2, block.x), CEIL(H / 2, block.y), C * batch);
    // maxpool2d_kernel<<<grid, block, 0, stream>>>(input, output, C, H, W);
    maxpool2d_kernel<<<grid, block>>>(input, output, C, H, W);
}

// 全连接层实现
// input:  [batch, in_dim]
// weight: [out_dim, in_dim]
// bias:   [out_dim]
// output: [batch, out_dim]
void linear_forward(const float *input, const float *weight, const float *bias,
                    float *output, int batch, int in_dim, int out_dim,
                    bool accumulate) {

    int M = batch;
    int K = in_dim;
    int N = out_dim;

    dim3 block(TILE_N / RN, TILE_M / RM); // (threads_per_row, threads_per_col)
    dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));
    tiled_gemm<<<grid, block>>>(input, weight, bias, output, M, K, N, accumulate);
}

// IFNode 激活层实现: V: [batch, N], I: [batch, N], S: [batch, N]
// V ← V + I
// if (V >= threshold) then S=1, V=0 else S=0
__global__ void IFNode_kernel(float *V, const float *I, float *S, int batch,
                              int N, float threshold) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= batch * N)
        return;

    float v = V[gid] + I[gid];
    if (v >= threshold) {
        S[gid] = 1.0f;
        v = 0.0f;
        // v -= threshold; // 减去阈值，保留多余电位
    } else {
        S[gid] = 0.0f;
    }
    V[gid] = v;
}

void IFNode_forward(float *V, const float *I, float *S, int batch, int N,
                    float threshold) {
    int threads = 256;
    int blocks = CEIL(batch * N, threads);
    // IFNode_kernel<<<blocks, threads, 0, stream>>>(V, I, S, N, threshold);
    IFNode_kernel<<<blocks, threads>>>(V, I, S, batch, N, threshold);
}

void fused_conv2d_ifnode_forward(const float *input, const float *weight,
                                const float *bias, float *V, float *S,
                                int batch, int Cin, int Cout, int H, int W,
                                int kernel_size, int padding,
                                float threshold) {

    int Hout = (H - kernel_size + 2 * padding) + 1;
    int Wout = (W - kernel_size + 2 * padding) + 1;

    dim3 block(TILE_W, TILE_H);
    dim3 grid(CEIL(Wout, TILE_W), CEIL(Hout, TILE_H), batch * CEIL(Cout, OUT_PER_BLOCK));

    fused_conv2d_ifnode<<<grid, block>>>(input, weight, bias, V, S, Cin, Cout,
                                         H, W, padding, Hout, Wout, threshold);
}

void fused_linear_ifnode_forward(const float *input, const float *weight,
                                const float *bias, float *V, float *S,
                                int batch, int in_dim, int out_dim,
                                float threshold) {
    int M = batch;
    int K = in_dim;
    int N = out_dim;

    dim3 block(TILE_N / RN, TILE_M / RM); // (threads_per_row, threads_per_col)
    dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));
    fused_gemm_ifnode_kernel<<<grid, block>>>(input, weight, bias, V, S, M, K, N, threshold);
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>"
                << std::endl;
        return 1;
    }
    std::string dir = argv[1];

    // Load test data
    auto images = read_mnist_images(
        dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(
        dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty())
        return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");

    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w, fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b, fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w, fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b, fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w, fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b, fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(),
                                conv1_w.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(),
                                conv1_b.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(),
                                conv2_w.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(),
                                conv2_b.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(),
                                fc1_w.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(),
                                fc1_b.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(),
                                fc2_w.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(),
                                fc2_b.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(),
                                fc3_w.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(),
                                fc3_b.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // ===================================================================================
    // Main Function -  DO NOT MODIFY END
    // ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
    // auto input_channels = conv1_w.size() / 9; // conv1: (C, 1, 3, 3)
    // SCNNConfig cfg = init_scnn_config(input_channels);
    std::vector<int> predictions =
        scnn_inference(images, d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
                        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b,
                        2048 // batch size
        );

    // ===================================================================================
    // Main Function -  DO NOT MODIFY BEGIN
    // ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));

    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();

    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":"
              << accuracy << std::endl;

    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================
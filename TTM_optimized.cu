#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

__global__ void ttm_mode1_optimized(float *X, float *M, float *Y, int A, int B, int C, int D) {
    __shared__ float tile_M[32][32];
    __shared__ float tile_X[32][32];

    int a = blockIdx.z;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    float val = 0.0f;

    for (int b0 = 0; b0 < B; b0 += 32) {
        if (d < D && (b0 + threadIdx.y) < B)
            tile_M[threadIdx.x][threadIdx.y] = M[d * B + b0 + threadIdx.y];
        else
            tile_M[threadIdx.x][threadIdx.y] = 0.0f;

        if (a < A && (b0 + threadIdx.x) < B && c < C)
            tile_X[threadIdx.x][threadIdx.y] = X[a * B * C + (b0 + threadIdx.x) * C + c];
        else
            tile_X[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        for (int b = 0; b < 32; ++b)
            val += tile_M[threadIdx.x][b] * tile_X[b][threadIdx.y];

        __syncthreads();
    }

    if (a < A && d < D && c < C)
        Y[a * D * C + d * C + c] = val;
}

int main() {
    const int A = 256, B = 512, C = 512, D = 256;
    size_t size_X = A * B * C * sizeof(float);
    size_t size_M = D * B * sizeof(float);
    size_t size_Y = A * D * C * sizeof(float);

    float *h_X = (float*)malloc(size_X);
    float *h_M = (float*)malloc(size_M);
    float *h_Y = (float*)malloc(size_Y);

    for (int i = 0; i < A * B * C; ++i)
        h_X[i] = static_cast<float>(i % 10);
    for (int i = 0; i < D * B; ++i)
        h_M[i] = (i % 3 == 0) ? 1.0f : 0.5f;

    float *d_X, *d_M, *d_Y;
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_M, size_M);
    cudaMalloc(&d_Y, size_Y);

    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockDim(32, 32);
    dim3 gridDim((D + 31) / 32, (C + 31) / 32, A);

    cudaFuncSetCacheConfig(ttm_mode1_optimized, cudaFuncCachePreferShared);

    cudaEventRecord(start);
    ttm_mode1_optimized<<<gridDim, blockDim>>>(d_X, d_M, d_Y, A, B, C, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_Y, d_Y, size_Y, cudaMemcpyDeviceToHost);

    float *cpu_Y = (float*)malloc(size_Y);
    for (int a = 0; a < A; ++a)
        for (int d = 0; d < D; ++d)
            for (int c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int b = 0; b < B; ++b)
                    sum += h_M[d * B + b] * h_X[a * B * C + b * C + c];
                cpu_Y[a * D * C + d * C + c] = sum;
            }

    bool all_good = true;
    for (int i = 0; i < A * D * C; ++i) {
        if (fabs(h_Y[i] - cpu_Y[i]) > 1e-3) {
            std::cout << "Mismatch at index " << i
                      << ": CPU = " << cpu_Y[i]
                      << ", GPU = " << h_Y[i] << "\n";
            all_good = false;
            break;
        }
    }
    if (all_good) std::cout << "✅ CUDA output matches CPU output!\n";
    else          std::cout << "❌ CUDA output has mismatches.\n";

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA Kernel execution time: " << milliseconds << " ms\n";
    std::cout << "Y[0][0][0] = " << h_Y[0] << "\n";

    free(h_X); free(h_M); free(h_Y); free(cpu_Y);
    cudaFree(d_X); cudaFree(d_M); cudaFree(d_Y);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

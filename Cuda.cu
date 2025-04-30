#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void ttm_mode1(float *X, float *M, float *Y, int A, int B, int C, int D) {
    int a = blockIdx.z;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (a < A && d < D && c < C) {
        float val = 0.0f;
        for (int b = 0; b < B; ++b) {
            int x_idx = a * B * C + b * C + c;
            int m_idx = d * B + b;
            val += M[m_idx] * X[x_idx];
        }
        int y_idx = a * D * C + d * C + c;
        Y[y_idx] = val;
    }
}

int main() {
    const int A = 256;
    const int B = 512;
    const int C = 512; 
    const int D = 256; 
    size_t size_X = A * B * C * sizeof(float);
    size_t size_M = D * B * sizeof(float);
    size_t size_Y = A * D * C * sizeof(float);

    float *h_X = (float*)malloc(size_X);
    float *h_M = (float*)malloc(size_M);
    float *h_Y = (float*)malloc(size_Y);

    for (int i = 0; i < A * B * C; ++i) h_X[i] = static_cast<float>(i % 10);
    for (int i = 0; i < D * B; ++i) h_M[i] = (i % 3 == 0) ? 1.0f : 0.5f;

    float *d_X, *d_M, *d_Y;
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_M, size_M);
    cudaMalloc(&d_Y, size_Y);

    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockDim(16, 16);
    dim3 gridDim((D + 15) / 16, (C + 15) / 16, A);

    cudaEventRecord(start);
    ttm_mode1<<<gridDim, blockDim>>>(d_X, d_M, d_Y, A, B, C, D);
    cudaEventRecord(stop);
 cudaEventSynchronize(stop);
    cudaMemcpy(h_Y, d_Y, size_Y, cudaMemcpyDeviceToHost);


float *cpu_Y = (float*)malloc(size_Y);
for (int a = 0; a < A; ++a)
    for (int d = 0; d < D; ++d)
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int b = 0; b < B; ++b) {
                sum += h_M[d * B + b] * h_X[a * B * C + b * C + c];
            }
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

free(cpu_Y);
   
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA Kernel execution time: " << milliseconds << " ms\n";

    std::cout << "Y[0][0][0] = " << h_Y[0] << "\n";

    free(h_X); free(h_M); free(h_Y);
    cudaFree(d_X); cudaFree(d_M); cudaFree(d_Y);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}



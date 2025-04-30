#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <cmath>

int main() {
    const int A = 256, B = 512, C = 512, D = 256;
    size_t size_X = A * B * C;
    size_t size_M = D * B;
    size_t size_Y = A * D * C;

    float *X = (float*)malloc(size_X * sizeof(float));
    float *M = (float*)malloc(size_M * sizeof(float));
    float *Y = (float*)malloc(size_Y * sizeof(float));


    for (int i = 0; i < size_X; ++i) X[i] = static_cast<float>(i % 10);
    for (int i = 0; i < size_M; ++i) M[i] = (i % 3 == 0) ? 1.0f : 0.5f;

    double start = omp_get_wtime();

    // TTM along mode-1
    #pragma omp parallel for collapse(3)
    for (int a = 0; a < A; ++a)
        for (int d = 0; d < D; ++d)
            for (int c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int b = 0; b < B; ++b) {
                    sum += M[d * B + b] * X[a * B * C + b * C + c];
                }
                Y[a * D * C + d * C + c] = sum;
            }

    double end = omp_get_wtime();
    std::cout << "OpenMP execution time: " << (end - start) * 1000 << " ms\n";
    std::cout << "Y[0][0][0] = " << Y[0] << "\n";

    free(X); free(M); free(Y);
    return 0;
}

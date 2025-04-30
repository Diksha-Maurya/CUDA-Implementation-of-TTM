#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int A = 256, B = 512, C = 512, D = 256;
    size_t size_X = A * B * C;
    size_t size_M = D * B;
    size_t size_Y = A * D * C;

    float *X = nullptr;
    float *M = nullptr;
    float *Y = nullptr;

    // Allocate and initialize on rank 0
    if (rank == 0) {
        X = (float*)malloc(size_X * sizeof(float));
        M = (float*)malloc(size_M * sizeof(float));
        Y = (float*)malloc(size_Y * sizeof(float));

        for (int i = 0; i < size_X; ++i) X[i] = static_cast<float>(i % 10);
        for (int i = 0; i < size_M; ++i) M[i] = (i % 3 == 0) ? 1.0f : 0.5f;
    }

    // Broadcast M to all processes
    if (rank != 0) M = (float*)malloc(size_M * sizeof(float));
    MPI_Bcast(M, D * B, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Split slices (A) among processes
    int slices_per_proc = A / size;
    int remainder = A % size;
    int local_A = slices_per_proc + (rank < remainder ? 1 : 0);

    int start_a = slices_per_proc * rank + std::min(rank, remainder);

    float* X_local = (float*)malloc(local_A * B * C * sizeof(float));
    float* Y_local = (float*)malloc(local_A * D * C * sizeof(float));

    // Scatter X manually
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            int local_Ap = slices_per_proc + (p < remainder ? 1 : 0);
            int start_ap = slices_per_proc * p + std::min(p, remainder);
            if (p == 0) {
                std::memcpy(X_local, X + start_ap * B * C, local_Ap * B * C * sizeof(float));
            } else {
                MPI_Send(X + start_ap * B * C, local_Ap * B * C, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(X_local, local_A * B * C, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int a = 0; a < local_A; ++a)
        for (int d = 0; d < D; ++d)
            for (int c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int b = 0; b < B; ++b) {
                    sum += M[d * B + b] * X_local[a * B * C + b * C + c];
                }
                Y_local[a * D * C + d * C + c] = sum;
            }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_duration = (end_time - start_time) * 1000;

    std::cout << "Rank " << rank << ": Execution time = " << local_duration << " ms\n";

    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int local_Ap = slices_per_proc + (p < remainder ? 1 : 0);
            int start_ap = slices_per_proc * p + std::min(p, remainder);
            MPI_Recv(Y + start_ap * D * C, local_Ap * D * C, MPI_FLOAT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        std::memcpy(Y, Y_local, local_A * D * C * sizeof(float));

        std::cout << "Y[0][0][0] = " << Y[0] << "\n";
        free(X); free(Y);
    } else {
        MPI_Send(Y_local, local_A * D * C, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }

    free(M);
    free(X_local); free(Y_local);
    MPI_Finalize();
    return 0;
}

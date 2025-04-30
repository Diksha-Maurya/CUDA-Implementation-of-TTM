# CUDA Implementation of TTM (Tensor-Times-Matrix)

## 📌 Project Description
This project implements the Tensor-Times-Matrix (TTM) operation using three parallel computing approaches: **CUDA**, **MPI**, and **OpenMP**. The TTM operation is a key component in tensor decompositions and is commonly used in machine learning, scientific computing, and multidimensional data analysis. Our goal is to compare the performance of these implementations and evaluate GPU acceleration benefits.

## ⚙️ Installation & Usage

### ✅ Prerequisites
Ensure you have:
- NVIDIA GPU with CUDA installed (for CUDA version)
- MPI installed (`mpicxx`, `mpirun`)
- GCC with OpenMP support

### 🚀 Compile & Run

#### 🔹 CUDA Implementation (Tiling Optimized)
```bash
nvcc -O3 -o ttm_optimized TTM_optimized.cu
./ttm_optimized
```

#### 🔹 CUDA Implementation
```bash
nvcc -O2 -o cuda1 Cuda.cu
./cuda1
```

#### 🔹 MPI Implementation
```bash
mpicxx -O2 -o mpi1 MPI.cpp
mpirun -np 4 ./mpi1
```

#### 🔹 OpenMP Implementation
```bash
g++ -fopenmp -O2 -o openmp OpenMP.cpp
./openmp
```

## 👩‍💻 Team Members
- Diksha Maurya  
*(North Carolina State University)*

## 📊 Performance Comparison

The chart below compares the execution time of different TTM Mode-1 implementations on a tensor of shape (256 × 512 × 512) and matrix (256 × 512). The tiling-optimized CUDA version significantly outperformed all others, completing in 343.221 ms, compared to 540.000 ms for the naive CUDA version. The MPI and OpenMP implementations were considerably slower, taking 1611.340 ms and 1811.380 ms, respectively. These results demonstrate the effectiveness of GPU acceleration, especially with shared memory and memory coalescing optimizations.


![ttm_time_comparison](https://github.com/user-attachments/assets/5b48ebb6-c940-4e41-a387-6a2b76236a09)



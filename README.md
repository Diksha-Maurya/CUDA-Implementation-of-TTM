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


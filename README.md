# Parallel-and-distributed-Computing

📘 README - MPI Program Collection
This repository includes various MPI-based parallel computing programs, each illustrating key MPI techniques such as point-to-point communication, collective communication, and parallel computation.

🖥️ System Requirements
Linux:

Run these commands in bash

sudo apt update

sudo apt install mpich

Windows:

1.Install MinGW-w64 compiler.

2.Install Microsoft MPI (MSMPI):

3.Download .exe and .msi files and install in directories without spaces.

4.Set environment variables:

MSMPI_INC and MSMPI_LIB64

5.Configure VSCode tasks.json:

"-I", "${MSMPI_INC}",

"-L", "${MSMPI_LIB64}",

"-lmsmpi"

Compilation Instructions


All MPI programs:

Run in bash

mpicc <source_file.c> -o <output_file>

Examples:

Run in bash

mpicc mpi_monte_carlo_pi.c -o monte_pi

mpicc mpi_matrix_multiply.c -o matrix_multiply

mpicc mpi_odd_even_sort.c -o odd_even_sort
...
🚀 Execution Instructions

For all programs:

Run in bash

mpirun -np <num_processes> ./<output_file>

Example:

Run in bash

mpirun -np 4 ./monte_pi

📄 Brief Description of Each Program

1️⃣ Monte Carlo Pi Estimation

Random points generated to estimate Pi.
Demonstrates random number generation, reduction of partial results.

2️⃣ Matrix Multiplication
Multiply two 70x70 matrices.
Compare Serial Execution Time vs Parallel MPI Time.
Uses omp_get_wtime() and MPI_Wtime() for timing.

3️⃣ Parallel Odd-Even Sort

Parallel sorting algorithm.
Uses process synchronization after each phase.

4️⃣ Heat Distribution Simulation

Simulates 2D grid heat flow.
Communication at grid boundaries between neighboring processes.

5️⃣ Parallel Reduction

Aggregates data from all processes (sum, max, etc.).
Uses MPI_Reduce().

6️⃣ Parallel Dot Product

Computes dot product of two large vectors.
Divides vector chunks among processes.

7️⃣ Parallel Prefix Sum (Scan)

Computes prefix sum (cumulative sum).
Uses MPI_Scan().

8️⃣ Parallel Matrix Transposition

Transposes a matrix across multiple processes.
Demonstrates send/receive of rows/columns.

9️⃣ DAXPY Operation

Performs vector operation X[i] = a * X[i] + Y[i].
Speedup comparison: Serial vs Parallel

🔟 Pi Calculation using MPI_Bcast & MPI_Reduce

Broadcasts total steps via MPI_Bcast.
Reduces partial Pi sums via MPI_Reduce.

🔢 Prime Finder using MPI_Send & MPI_Recv

Master-slave approach:
Master distributes numbers.
Slaves test numbers and send results back.
Uses MPI_ANY_SOURCE.

13.Task sum threads

# CUDA Program: Threads Performing Different Tasks

This CUDA program demonstrates how **individual GPU threads can be assigned different computational tasks**. In this example, we perform **two methods to calculate the sum of the first `N` integers (N = 1024)**.

---

## 🎯 Objective

The program launches **2 threads**:

- **Thread 0**: Calculates the sum of the first `N` integers using an **iterative approach**.

- **Thread 1**: Calculates the sum using the **mathematical formula**:  
  \[
  \text{Sum} = \frac{N(N + 1)}{2}
  \]

---

## 🔧 Files

| File Name         | Description                              |
|------------------|------------------------------------------|
| `cuda_sum_tasks.cu` | CUDA source file where threads do separate work |

---

## 📦 Requirements

To run this code, you need:

- A system with **NVIDIA GPU** and **CUDA toolkit** installed  

- Or use **Google Colab** with GPU enabled

---

## 🚀 How to Run in Google Colab

### 1. Make sure GPU is enabled:

`Runtime` > `Change runtime type` > Set **Hardware Accelerator** to `GPU`

### 2. Save the CUDA code

```cpp
%%writefile cuda_sum_tasks.cu

// Paste the CUDA code here

Compile and Run

!nvcc cuda_sum_tasks.cu -o cuda_sum_tasks

!./cuda_sum_tasks

14.Mergesort

# CUDA Bitonic Sort

This project implements **Bitonic Sort** using CUDA. Bitonic sort is a parallel sorting algorithm ideal for GPU computation due to its predictable memory access and computation patterns.

---

## 🔧 Files

| File Name         | Description                            |
|------------------|----------------------------------------|
| `bitonic_sort.cu` | CUDA implementation of Bitonic sort   |

---

## 📌 Requirements

- NVIDIA GPU with CUDA support

- CUDA Toolkit installed  

OR  

- Use **Google Colab** with GPU enabled

---

## ⚙️ How It Works

- Sorts an array of integers using the **Bitonic Sorting Network**.

- Uses nested CUDA kernel launches to perform bitonic merges and sorting stages.

- Measures and prints execution time using `std::chrono`.

---

## 📥 How to Run (Google Colab)

### 1. Enable GPU in Colab

> Go to `Runtime` → `Change runtime type` → Set **Hardware accelerator** to `GPU`

---

### 2. Save the Code

Paste the code in a Colab cell with:

```cpp

%%writefile bitonic_sort.cu

// [ Paste your CUDA Bitonic sort code here ]

Compile and Run

!nvcc -O3 bitonic_sort.cu -o bitonic_sort

!./bitonic_sort


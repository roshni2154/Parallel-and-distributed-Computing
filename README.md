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
Speedup comparison: Serial vs Parallel.
🔟 Pi Calculation using MPI_Bcast & MPI_Reduce
Broadcasts total steps via MPI_Bcast.
Reduces partial Pi sums via MPI_Reduce.
🔢 Prime Finder using MPI_Send & MPI_Recv
Master-slave approach:
Master distributes numbers.
Slaves test numbers and send results back.
Uses MPI_ANY_SOURCE.

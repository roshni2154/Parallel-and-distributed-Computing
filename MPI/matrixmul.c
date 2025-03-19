#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0

// Function to multiply matrices
void multiply(int N, double *A, double *B, double *C, int start, int end) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size, N = 1000;
    double *A, *B, *C;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Master initializes matrices
    if (rank == MASTER) {
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));

        // Initialize matrices A and B
        for (int i = 0; i < N * N; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
            C[i] = 0.0;
        }
    } else {
        // Other processes allocate memory for partial results
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // Scatter rows of matrix A to all processes
    int rows_per_process = N / size;
    double *sub_A = (double *)malloc(rows_per_process * N * sizeof(double));
    double *sub_C = (double *)malloc(rows_per_process * N * sizeof(double));

    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE, sub_A, rows_per_process * N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Perform multiplication on allocated rows
    multiply(N, sub_A, B, sub_C, 0, rows_per_process);

    // Gather results from all processes
    MPI_Gather(sub_C, rows_per_process * N, MPI_DOUBLE, C, rows_per_process * N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    // Master prints the result and time taken
    if (rank == MASTER) {
        printf("Matrix Size: %d x %d, Time: %f seconds\n", N, N, end - start);
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);
    free(sub_A);
    free(sub_C);

    MPI_Finalize();
    return 0;
}
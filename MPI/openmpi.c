#include <mpi.h>
#include <stdio.h>
#include<string.h>
int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
// Get the number of processes
// Print off a hello world message
     printf("Hello world from processor "
           );

    // Finalize the MPI environment.
    MPI_Finalize();
}
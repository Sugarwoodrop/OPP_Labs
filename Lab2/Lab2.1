#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 30000

double Norma(double* Vector, int size) {
    double local_sum = 0, global_sum;
    for (int i = 0; i < size; ++i) {
        local_sum += Vector[i] * Vector[i];
    }
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(global_sum);
}

void Multiplication(double* final_vector, double* Matrix, double* Vector, int rows_per_proc) {
    for (int i = 0; i < rows_per_proc; ++i) {
        double sum = 0;
        for (int j = 0; j < N; ++j) { 
            sum += Matrix[i * N + j] * Vector[j];
        }
        final_vector[i] = sum;
    }
}

void Minus(double* result, double* first_vector, double* second_vector, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = first_vector[i] - second_vector[i];
    }
}

void Multiplication_Scalar(double* Vector, int size) {
    for (int i = 0; i < size; ++i) {
        Vector[i] *= 0.01;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    int extra_rows = N % size;
    int Lacal_Rows = (rank < extra_rows) ? (rows_per_proc + 1) : rows_per_proc;

    int startRowGlobalIndex = rank * rows_per_proc + ((rank < extra_rows) ? rank : extra_rows);

    double* local_matrix = (double*)calloc(Lacal_Rows * N, sizeof(double));
    double* desired_vector = (double*)calloc(N, sizeof(double));
    double* arbitrary_vector = (double*)calloc(N, sizeof(double));

    if (!local_matrix || !desired_vector || !arbitrary_vector) {
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        arbitrary_vector[i] = sin(2 * 3.14159 * i / N);
    }

    for (int i = 0; i < Lacal_Rows; ++i) {
        int globalRow = startRowGlobalIndex + i;
        for (int j = 0; j < N; ++j) {
            local_matrix[i * N + j] = (globalRow == j) ? 2.0 : 1.0;
        }
    }

    double* vector = (double*)calloc(Lacal_Rows, sizeof(double));
    double* mult = (double*)calloc(Lacal_Rows, sizeof(double));
    double* final = (double*)calloc(Lacal_Rows, sizeof(double));

    double t1 = MPI_Wtime();

    Multiplication(vector, local_matrix, arbitrary_vector, Lacal_Rows);
    Multiplication(mult, local_matrix, desired_vector, Lacal_Rows);
    Minus(final, mult, vector, Lacal_Rows);

    double norm_vector = Norma(vector, Lacal_Rows);
    double norm_final = Norma(final, Lacal_Rows);

    while ((norm_final / norm_vector) > 1e-5) {
        Multiplication_Scalar(final, Lacal_Rows);

        for (int i = 0; i < Lacal_Rows; i++) {
            desired_vector[startRowGlobalIndex + i] -= final[i];
        }

        MPI_Allgather(MPI_IN_PLACE, Lacal_Rows, MPI_DOUBLE,
                      desired_vector, Lacal_Rows, MPI_DOUBLE, MPI_COMM_WORLD);

        Multiplication(mult, local_matrix, desired_vector, Lacal_Rows);
        Minus(final, mult, vector, Lacal_Rows);
        norm_final = Norma(final, Lacal_Rows);
    }

    double t2 = MPI_Wtime();

    if (rank == 0) {
        printf("Time: %f seconds\n", t2 - t1);
    }

    free(local_matrix);
    free(desired_vector);
    free(vector);
    free(mult);
    free(final);

    MPI_Finalize();
    return 0;
}

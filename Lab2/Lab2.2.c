#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#define N 30000

double Norma(double* Vector, int size) {
	double sum = 0, local_sum = 0, final_number;

    for (int i = 0; i < size; ++i) {
        local_sum += Vector[i] * Vector[i];
    }

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    final_number = sqrt(sum);

    return final_number;
}

void Multiplication(double* final_vector, double* Matrix, double* Vector, int rows_per_proc, int rank, int num_procs) {
    for (int i = 0; i < rows_per_proc; ++i) {
        final_vector[i] = 0.0;
    }

    int max_rows_per_proc = (N + num_procs - 1) / num_procs;
    double* sendBuf = (double*)malloc(max_rows_per_proc * sizeof(double));
    double* recvBuf = (double*)malloc(max_rows_per_proc * sizeof(double));

    memcpy(sendBuf, Vector, rows_per_proc * sizeof(double));
    int currentOwner = rank;
    int currentSize = rows_per_proc;

    for (int step = 0; step < num_procs; ++step) {
        int sendSize = currentSize;  // Запоминаем размер перед отправкой

        for (int i = 0; i < rows_per_proc; ++i) {
            double sum = 0.0;
            for (int j = 0; j < sendSize; ++j) { // Используем sendSize, а не обновлённый currentSize
                int globalIndex = currentOwner * rows_per_proc + j;
                sum += Matrix[i * N + globalIndex] * sendBuf[j];
            }
            final_vector[i] += sum;
        }

        int leftNeighbour = (rank - 1 + num_procs) % num_procs;
        int rightNeighbour = (rank + 1) % num_procs;
        int sendMeta[2] = {sendSize, currentOwner};
        int recvMeta[2];

        MPI_Sendrecv(sendMeta, 2, MPI_INT, rightNeighbour, 0,
                     recvMeta, 2, MPI_INT, leftNeighbour, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int nextSize = recvMeta[0];
        int nextOwner = recvMeta[1];

        MPI_Sendrecv(sendBuf, sendSize, MPI_DOUBLE, rightNeighbour, 1,
                     recvBuf, nextSize, MPI_DOUBLE, leftNeighbour, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        currentSize = nextSize;   // Обновляем currentSize только после успешного получения
        currentOwner = nextOwner;
        memcpy(sendBuf, recvBuf, currentSize * sizeof(double));
    }

    free(sendBuf);
    free(recvBuf);
}


void Minus(double* result,double* first_vector, double* second_vector, int size) {
	for (int i = 0; i < size; ++i) {
        result[i] = first_vector[i] - second_vector[i];
    }
}

void Multiplication_Scalar(double* Vector, int size) {
	for (int i = 0; i < size; ++i) {
        Vector[i] = Vector[i] * 0.01;
    }
}


int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	  int rows_per_proc = N / size;         // Базовое число строк в каждом процессе
    int extra_rows = N % size; 			  // Остаток строк, которые надо распределить

    int Lacal_Rows = (rank < extra_rows) ? (rows_per_proc + 1) : rows_per_proc;

    int startRowGlobalIndex = 0;
    for (int r = 0; r < rank; ++r) {
        startRowGlobalIndex += (r < extra_rows) ? (rows_per_proc + 1) : rows_per_proc;
    }

    double* local_matrix = (double*)calloc(Lacal_Rows * N, sizeof(double));
    double* desired_vector = (double*)calloc(Lacal_Rows, sizeof(double));
    double* arbitrary_vector = (double*)calloc(Lacal_Rows, sizeof(double));
    double* vector = (double*)calloc(Lacal_Rows , sizeof(double)); //b
    double* mult = (double*)calloc(Lacal_Rows , sizeof(double)); //Ax
    double* final = (double*)calloc(Lacal_Rows , sizeof(double)); // Ax - b

	if (!local_matrix || !desired_vector || !arbitrary_vector) {
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < Lacal_Rows; ++i) {
         int globalIdx = startRowGlobalIndex + i;
         arbitrary_vector[i] = sin(2 * 3.14159 * globalIdx / N);
    }

    for (int i = 0; i < Lacal_Rows; ++i) {
        int globalRow = startRowGlobalIndex + i;
        for (int j = 0; j < N; ++j) {
            if (globalRow == j)
                local_matrix[i * N + j] = 2.0;
            else
                local_matrix[i * N + j] = 1.0;
        }
    }
    

    double t1 = MPI_Wtime();

	Multiplication(vector, local_matrix, arbitrary_vector, Lacal_Rows, rank, size); //b
    Multiplication(mult, local_matrix, desired_vector, Lacal_Rows, rank, size); //Ax
    Minus(final, mult, vector, Lacal_Rows); // Ax - b

    double norm_vector = Norma(vector, Lacal_Rows); // ||b||
    double norm_final = Norma(final, Lacal_Rows); // ||Ax - b||

	while ((norm_final / norm_vector) > 1e-5) {
		Multiplication_Scalar(final, Lacal_Rows);

		  for (int i = 0; i < Lacal_Rows; i++) {
			  desired_vector[i] -= final[i];
		  }

        Multiplication(mult, local_matrix, desired_vector, Lacal_Rows, rank, size);
        Minus(final, mult, vector, Lacal_Rows);

        norm_final = Norma(final, Lacal_Rows);
	}
    
    double t2 = MPI_Wtime();

	if (rank == 0) {
    printf("%f\n", t2 - t1);
  }
	free(local_matrix);
	free(desired_vector);
	free(vector);
	free(mult);
	free(final);
	MPI_Finalize(); 
	return 0;
}

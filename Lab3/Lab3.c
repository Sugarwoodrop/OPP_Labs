#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define N1 1600 
#define N2 1600 
#define N3 1600 

double* allocateMatrix1D(int size) {
    double* matrix = (double*)malloc(size * sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "Ошибка выделения памяти для матрицы\n");
        exit(EXIT_FAILURE);
    }
    return matrix;
}

void initIdentityMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void initRandomMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = 1.0 + rand() % 10;
        }
    }
}

void printMatrix(double* matrix, int rows, int cols, const char* name) {
    printf("Матрица %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm grid_comm;
    int dims[2], periods[2], coords[2];
    int p1, p2; // Размеры решетки процессов
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    srand(time(NULL) + rank);
    
    dims[0] = dims[1] = 0;
    MPI_Dims_create(size, 2, dims);
    p1 = dims[0];
    p2 = dims[1];
    
    if (rank == 0) {
        printf("Создание решетки процессов %dx%d для матриц размером (%dx%d)x(%dx%d)\n", 
               p1, p2, N1, N2, N2, N3);
    }
    
    periods[0] = periods[1] = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int row_coord = coords[0];
    int col_coord = coords[1];
    
    int local_n1 = N1 / p1 + (row_coord < (N1 % p1) ? 1 : 0);
    int local_n3 = N3 / p2 + (col_coord < (N3 % p2) ? 1 : 0);
    
    MPI_Comm row_comm, col_comm;
    int remain_dims[2];
    
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(grid_comm, remain_dims, &row_comm);
    
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(grid_comm, remain_dims, &col_comm);
    
    double* local_A = allocateMatrix1D(local_n1 * N2);
    double* local_B = allocateMatrix1D(N2 * local_n3);
    double* local_C = allocateMatrix1D(local_n1 * local_n3);
    
    for (int i = 0; i < local_n1 * local_n3; i++) {
        local_C[i] = 0.0;
    }
    
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    
    if (rank == 0) {
        A = allocateMatrix1D(N1 * N2);
        B = allocateMatrix1D(N2 * N3);
        C = allocateMatrix1D(N1 * N3);
        
        initIdentityMatrix(A, N1, N2);
        
        initRandomMatrix(B, N2, N3);
    }
    
    if (col_coord == 0) {
        int* sendcounts_A = NULL;
        int* displs_A = NULL;
        
        if (rank == 0) {
            sendcounts_A = (int*)calloc(p1, sizeof(int));
            displs_A = (int*)calloc(p1, sizeof(int));
            
            for (int i = 0; i < p1; i++) {
                int rows_in_block = N1 / p1 + (i < (N1 % p1) ? 1 : 0);
                sendcounts_A[i] = rows_in_block * N2;
                displs_A[i] = (i > 0) ? (displs_A[i-1] + sendcounts_A[i-1]) : 0;
            }
        }
        
        MPI_Scatterv(A, sendcounts_A, displs_A, MPI_DOUBLE, 
                     local_A, local_n1 * N2, MPI_DOUBLE, 
                     0, col_comm);
        
        if (rank == 0) {
            free(sendcounts_A);
            free(displs_A);
        }
    }
    
    if (row_coord == 0) {
        int* sendcounts_B = NULL;
        int* displs_B = NULL;
        double* temp_B = NULL;
        
        if (rank == 0) {
            sendcounts_B = (int*)calloc(p2, sizeof(int));
            displs_B = (int*)calloc(p2, sizeof(int));
            temp_B = (double*)malloc(N2 * N3 * sizeof(double));
            
            for (int i = 0; i < N2; i++) {
                for (int j = 0; j < N3; j++) {
                    temp_B[j * N2 + i] = B[i * N3 + j];
                }
            }
            
            for (int j = 0; j < p2; j++) {
                int cols_in_block = N3 / p2 + (j < (N3 % p2) ? 1 : 0);
                sendcounts_B[j] = cols_in_block * N2;
                displs_B[j] = (j > 0) ? (displs_B[j-1] + sendcounts_B[j-1]) : 0;
            }
        }
        
        
        MPI_Scatterv(temp_B, sendcounts_B, displs_B, MPI_DOUBLE, 
                     local_B, local_n3 * N2, MPI_DOUBLE, 
                     0, row_comm);
        
        
        
        if (rank == 0) {
            free(sendcounts_B);
            free(displs_B);
            free(temp_B);
        }
        
    }
    
    MPI_Bcast(local_A, local_n1 * N2, MPI_DOUBLE, 0, row_comm);
    
    MPI_Bcast(local_B, N2 * local_n3, MPI_DOUBLE, 0, col_comm);
    
    for (int i = 0; i < local_n1; i++) {
    for (int j = 0; j < local_n3; j++) {
        double sum = 0.0;
        for (int k = 0; k < N2; k++) {
            sum += local_A[i * N2 + k] * local_B[j * N2 + k];
        }
        local_C[i * local_n3 + j] = sum;
    }
}


  int local_size = local_n1 * local_n3;
  int* recvcounts = NULL;
  int* displs = NULL;

  if (rank == 0) {
    recvcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));
    
    int current_displ = 0;
    for (int proc = 0; proc < size; proc++) {
        int proc_coords[2];
        MPI_Cart_coords(grid_comm, proc, 2, proc_coords);
        
        int proc_row = proc_coords[0];
        int proc_col = proc_coords[1];
        
        int proc_n1 = N1 / p1 + (proc_row < (N1 % p1) ? 1 : 0);
        int proc_n3 = N3 / p2 + (proc_col < (N3 % p2) ? 1 : 0);
        
        recvcounts[proc] = proc_n1 * proc_n3;
        displs[proc] = current_displ;
        current_displ += recvcounts[proc];
    }
  }

  double* local_C_linear = (double*)malloc(local_size * sizeof(double));
  for (int i = 0; i < local_n1; i++) {
    for (int j = 0; j < local_n3; j++) {
        local_C_linear[i * local_n3 + j] = local_C[i * local_n3 + j];
    }
  }

  double* gathered_data = NULL;
  if (rank == 0) {
    gathered_data = (double*)malloc(N1 * N3 * sizeof(double));
  }

  MPI_Gatherv(local_C_linear, local_size, MPI_DOUBLE,
            gathered_data, recvcounts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < N1 * N3; i++) {
        C[i] = 0.0;
    }
    
    int data_index = 0;
    for (int proc = 0; proc < size; proc++) {
        int proc_coords[2];
        MPI_Cart_coords(grid_comm, proc, 2, proc_coords);
        
        int proc_row = proc_coords[0];
        int proc_col = proc_coords[1];
        
        int proc_n1 = N1 / p1 + (proc_row < (N1 % p1) ? 1 : 0);
        int proc_n3 = N3 / p2 + (proc_col < (N3 % p2) ? 1 : 0);
        int row_start = (N1 / p1) * proc_row + (proc_row < (N1 % p1) ? proc_row : (N1 % p1));
        int col_start = (N3 / p2) * proc_col + (proc_col < (N3 % p2) ? proc_col : (N3 % p2));
        
        for (int i = 0; i < proc_n1; i++) {
            for (int j = 0; j < proc_n3; j++) {
                int global_row = row_start + i;
                int global_col = col_start + j;
                C[global_row * N3 + global_col] = gathered_data[data_index++];
            }
        }
    }
        
        int equal = 1;
        for (int i = 0; i < N2 && equal; i++) {
          for (int j = 0; j < N3; j++) {
            if (fabs(C[i* N2 + j] - B[i * N3 + j]) > 1e-6) {
              equal = 0;
              break;
            }
          }
        }
        if (equal) {
          printf("Матрицы B и C совпадают\n");
        } else {
          printf("Матрицы B и C НЕ совпадают\n");
        }
        
        free(A);
        free(B);
        free(C);
        free(recvcounts);
        free(displs);
    }
    
  
    free(local_A);
    free(local_B);
    free(local_C);
    free(local_C_linear);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}

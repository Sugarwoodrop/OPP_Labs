#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define N1 3000 // Количество строк в матрице A и C
#define N2 3000 // Количество столбцов в A и строк в B
#define N3 3000 // Количество столбцов в матрицах B и C

double** allocateMatrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) {
        fprintf(stderr, "Ошибка выделения памяти для матрицы\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

void freeMatrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void initIdentityMatrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void initRandomMatrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 1.0 + rand() % 10;
        }
    }
}

void printMatrix(double** matrix, int rows, int cols, const char* name) {
    printf("Матрица %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i][j]);
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
      
    // Создание декартовой топологии
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
    
    double** local_A = allocateMatrix(local_n1, N2);
    double** local_B = allocateMatrix(N2, local_n3);
    double** local_C = allocateMatrix(local_n1, local_n3);
    
    for (int i = 0; i < local_n1; i++) {
        for (int j = 0; j < local_n3; j++) {
            local_C[i][j] = 0.0;
        }
    }
    
    double** A = NULL;
    double** B = NULL;
    double** C = NULL;
    
    if (rank == 0) {
        A = allocateMatrix(N1, N2);
        B = allocateMatrix(N2, N3);
        C = allocateMatrix(N1, N3);
        
        initIdentityMatrix(A, N1, N2);
        
        initRandomMatrix(B, N2, N3);
    }
    
    double t1 = MPI_Wtime();
    
    double *send_buf_A = NULL, *send_buf_B = NULL;
    if (rank == 0) {
        send_buf_A = (double*)malloc(N1 * N2 * sizeof(double));
        send_buf_B = (double*)malloc(N2 * N3 * sizeof(double));
        
        for (int i = 0; i < N1; i++) {
            for (int j = 0; j < N2; j++) {
                send_buf_A[i * N2 + j] = A[i][j];
            }
        }
        
        for (int i = 0; i < N2; i++) {
            for (int j = 0; j < N3; j++) {
                send_buf_B[i * N3 + j] = B[i][j];
            }
        }
    }
    
    double *row_buf_A = (double*)malloc(local_n1 * N2 * sizeof(double));
    double *col_buf_B = (double*)malloc(N2 * local_n3 * sizeof(double));

    
    if (col_coord == 0) {
        int *sendcounts_A = NULL, *displs_A = NULL;
        
        if (rank == 0) {
            sendcounts_A = (int*)calloc(p1, sizeof(int)); // сколько элементов отправить каждому процессу
            displs_A = (int*)calloc(p1, sizeof(int)); // указывает смещения от начала
            
            for (int i = 0; i < p1; i++) {
                int rows_in_block = N1 / p1 + (i < (N1 % p1) ? 1 : 0);
                sendcounts_A[i] = rows_in_block * N2;
                displs_A[i] = (i > 0) ? (displs_A[i-1] + sendcounts_A[i-1]) : 0;
            }
        }
        
        MPI_Scatterv(send_buf_A, sendcounts_A, displs_A, MPI_DOUBLE, 
                     row_buf_A, local_n1 * N2, MPI_DOUBLE, 
                     0, col_comm);
        
        if (rank == 0) {
            free(sendcounts_A);
            free(displs_A);
        }
    }
    
    if (row_coord == 0) {
        MPI_Datatype column_type;
        MPI_Type_vector(N2, local_n3, N3, MPI_DOUBLE, &column_type);
        MPI_Type_commit(&column_type);
        
        int *sendcounts_B = NULL, *displs_B = NULL;
        
        if (rank == 0) {
            sendcounts_B = (int*)calloc(p2, sizeof(int));
            displs_B = (int*)calloc(p2, sizeof(int));
            
            for (int j = 0; j < p2; j++) {
                sendcounts_B[j] = 1;
                displs_B[j] = (j > 0) ? (displs_B[j-1] + (N3 / p2) + (j-1 < (N3 % p2) ? 1 : 0)) : 0;
            }
        }
        
        if (rank == 0) {
            for (int j = 0; j < p2; j++) {
                int dest_rank;
                int dest_coords[2] = {0, j};
                MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
                
                if (dest_rank == 0) {
                    for (int i = 0; i < N2; i++) {
                        for (int k = 0; k < local_n3; k++) {
                            col_buf_B[i * local_n3 + k] = B[i][k];
                        }
                    }
                } else {
                    int cols_in_block = N3 / p2 + (j < (N3 % p2) ? 1 : 0);
                    int col_offset = (N3 / p2) * j + (j < (N3 % p2) ? j : (N3 % p2));
                    
                    double* temp_buf = (double*)malloc(N2 * cols_in_block * sizeof(double));
                    
                    for (int i = 0; i < N2; i++) {
                        for (int k = 0; k < cols_in_block; k++) {
                            temp_buf[i * cols_in_block + k] = B[i][col_offset + k];
                        }
                    }
                    
                    MPI_Send(temp_buf, N2 * cols_in_block, MPI_DOUBLE, dest_rank, 0, grid_comm);
                    
                    free(temp_buf);
                }
            }
        } else if (col_coord != 0) {
            MPI_Recv(col_buf_B, N2 * local_n3, MPI_DOUBLE, 0, 0, grid_comm, MPI_STATUS_IGNORE);// Прием данных
        }
        
        if (rank == 0) {
            free(sendcounts_B);
            free(displs_B);
        }
        
        MPI_Type_free(&column_type);
    }
    
    MPI_Bcast(row_buf_A, local_n1 * N2, MPI_DOUBLE, 0, row_comm);
    
    MPI_Bcast(col_buf_B, N2 * local_n3, MPI_DOUBLE, 0, col_comm);
    
    for (int i = 0; i < local_n1; i++) {
        for (int j = 0; j < N2; j++) {
            local_A[i][j] = row_buf_A[i * N2 + j];
        }
    }
    
    for (int i = 0; i < N2; i++) {
        for (int j = 0; j < local_n3; j++) {
            local_B[i][j] = col_buf_B[i * local_n3 + j];
        }
    }
    
    for (int i = 0; i < local_n1; i++) {
        for (int j = 0; j < local_n3; j++) {
            local_C[i][j] = 0.0;
            for (int k = 0; k < N2; k++) {
                local_C[i][j] += local_A[i][k] * local_B[k][j];
            }
        }
    }
    
    double *recv_buf_C = NULL;
    if (rank == 0) {
        recv_buf_C = (double*)malloc(N1 * N3 * sizeof(double));
    }
    
    double *local_C_buf = (double*)malloc(local_n1 * local_n3 * sizeof(double));
    for (int i = 0; i < local_n1; i++) {
        for (int j = 0; j < local_n3; j++) {
            local_C_buf[i * local_n3 + j] = local_C[i][j];
        }
    }
    
    if (rank == 0) {
        for (int i = 0; i < local_n1; i++) {
            for (int j = 0; j < local_n3; j++) {
                recv_buf_C[i * N3 + j] = local_C[i][j];
            }
        }
        
        for (int proc = 1; proc < size; proc++) {
            int proc_coords[2];
            MPI_Cart_coords(grid_comm, proc, 2, proc_coords);
            
            int proc_row = proc_coords[0];
            int proc_col = proc_coords[1];
            
            int proc_n1 = N1 / p1 + (proc_row < (N1 % p1) ? 1 : 0);
            int proc_n3 = N3 / p2 + (proc_col < (N3 % p2) ? 1 : 0);
            
            int proc_row_offset = (N1 / p1) * proc_row + (proc_row < (N1 % p1) ? proc_row : (N1 % p1));
            int proc_col_offset = (N3 / p2) * proc_col + (proc_col < (N3 % p2) ? proc_col : (N3 % p2));
            
            double *temp_buf = (double*)malloc(proc_n1 * proc_n3 * sizeof(double));
            MPI_Recv(temp_buf, proc_n1 * proc_n3, MPI_DOUBLE, proc, 0, grid_comm, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < proc_n1; i++) {
                for (int j = 0; j < proc_n3; j++) {
                    recv_buf_C[(proc_row_offset + i) * N3 + (proc_col_offset + j)] = temp_buf[i * proc_n3 + j];
                }
            }
            
            free(temp_buf);
        }
    } else {
        MPI_Send(local_C_buf, local_n1 * local_n3, MPI_DOUBLE, 0, 0, grid_comm);
    }
    
    if (rank == 0) {
        for (int i = 0; i < N1; i++) {
            for (int j = 0; j < N3; j++) {
                C[i][j] = recv_buf_C[i * N3 + j];
            }
        }
        double t2 = MPI_Wtime();
        printf("%f\n", t2 - t1);
        
        int equal = 1;
        for (int i = 0; i < N2 && equal; i++) {
          for (int j = 0; j < N3; j++) {
            if (fabs(C[i][j] - B[i][j]) > 1e-6) {
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
        
        freeMatrix(A, N1);
        freeMatrix(B, N2);
        freeMatrix(C, N1);
        free(send_buf_A);
        free(send_buf_B);
        free(recv_buf_C);
    }
    
    freeMatrix(local_A, local_n1);
    freeMatrix(local_B, N2);
    freeMatrix(local_C, local_n1);
    free(row_buf_A);
    free(col_buf_B);
    free(local_C_buf);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}

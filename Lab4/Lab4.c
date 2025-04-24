#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define EPSILON 1e-8
#define PARAM_A 1e5

#define X_MIN -1.0
#define Y_MIN -1.0
#define Z_MIN -1.0

#define X_SIZE 2.0
#define Y_SIZE 2.0
#define Z_SIZE 2.0

#define NX 400
#define NY 400
#define NZ 400

#define DX (X_SIZE / (NX - 1.0))
#define DY (Y_SIZE / (NY - 1.0))
#define DZ (Z_SIZE / (NZ - 1.0))

#define IDX3D(x, y, z) ((z) * NX * NY + (y) * NX + (x))

double exactSolution(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double sourceTerm(double x, double y, double z) {
    return 6 - PARAM_A * exactSolution(x, y, z);
}

void updateInteriorPoints(int procRank, int sliceHeight, double* oldValues, double* newValues, 
                        char* converged, double factor) {
    for (int z = 1; z < sliceHeight - 1; ++z) {
        for (int y = 1; y < NY - 1; ++y) {
            for (int x = 1; x < NX - 1; ++x) {
                double xTerm = (oldValues[IDX3D(x-1, y, z)] + oldValues[IDX3D(x+1, y, z)]) / (DX * DX);
                double yTerm = (oldValues[IDX3D(x, y-1, z)] + oldValues[IDX3D(x, y+1, z)]) / (DY * DY);
                double zTerm = (oldValues[IDX3D(x, y, z-1)] + oldValues[IDX3D(x, y, z+1)]) / (DZ * DZ);
                
                double xCoord = X_MIN + x * DX;
                double yCoord = Y_MIN + y * DY;
                double zCoord = Z_MIN + (z + sliceHeight * procRank) * DZ;
                
                newValues[IDX3D(x, y, z)] = factor * (xTerm + yTerm + zTerm - sourceTerm(xCoord, yCoord, zCoord));
                
                if (fabs(newValues[IDX3D(x, y, z)] - oldValues[IDX3D(x, y, z)]) > EPSILON) {
                    *converged = 0;
                }
            }
        }
    }
}

void updateBottomLayer(int procRank, int sliceHeight, double* oldValues, double* newValues, 
                     double* bottomGhost, char* converged, double factor) {
    if (procRank == 0) return;
    
    for (int y = 1; y < NY - 1; ++y) {
        for (int x = 1; x < NX - 1; ++x) {
            double xTerm = (oldValues[IDX3D(x-1, y, 0)] + oldValues[IDX3D(x+1, y, 0)]) / (DX * DX);
            double yTerm = (oldValues[IDX3D(x, y-1, 0)] + oldValues[IDX3D(x, y+1, 0)]) / (DY * DY);
            double zTerm = (bottomGhost[y*NX + x] + oldValues[IDX3D(x, y, 1)]) / (DZ * DZ);
            
            double xCoord = X_MIN + x * DX;
            double yCoord = Y_MIN + y * DY;
            double zCoord = Z_MIN + (sliceHeight * procRank) * DZ;
            
            newValues[IDX3D(x, y, 0)] = factor * (xTerm + yTerm + zTerm - sourceTerm(xCoord, yCoord, zCoord));
            
            if (fabs(newValues[IDX3D(x, y, 0)] - oldValues[IDX3D(x, y, 0)]) > EPSILON) {
                *converged = 0;
            }
        }
    }
}

void updateTopLayer(int procRank, int sliceHeight, double* oldValues, double* newValues, 
                  double* topGhost, char* converged, int numProcs, double factor) {
    if (procRank == numProcs - 1) return;
    
    int z = sliceHeight - 1;
    for (int y = 1; y < NY - 1; ++y) {
        for (int x = 1; x < NX - 1; ++x) {
            double xTerm = (oldValues[IDX3D(x-1, y, z)] + oldValues[IDX3D(x+1, y, z)]) / (DX * DX);
            double yTerm = (oldValues[IDX3D(x, y-1, z)] + oldValues[IDX3D(x, y+1, z)]) / (DY * DY);
            double zTerm = (oldValues[IDX3D(x, y, z-1)] + topGhost[y*NX + x]) / (DZ * DZ);
            
            double xCoord = X_MIN + x * DX;
            double yCoord = Y_MIN + y * DY;
            double zCoord = Z_MIN + (z + sliceHeight * procRank) * DZ;
            
            newValues[IDX3D(x, y, z)] = factor * (xTerm + yTerm + zTerm - sourceTerm(xCoord, yCoord, zCoord));
            
            if (fabs(newValues[IDX3D(x, y, z)] - oldValues[IDX3D(x, y, z)]) > EPSILON) {
                *converged = 0;
            }
        }
    }
}

double computeMaxError(int procRank, int sliceHeight, double* solution) {
    double localMaxError = 0.0;
    double error;
    
    for (int z = 0; z < sliceHeight; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                double xCoord = X_MIN + x * DX;
                double yCoord = Y_MIN + y * DY;
                double zCoord = Z_MIN + (z + sliceHeight * procRank) * DZ;
                
                error = fabs(solution[IDX3D(x, y, z)] - exactSolution(xCoord, yCoord, zCoord));
                
                if (error > localMaxError) {
                    localMaxError = error;
                }
            }
        }
    }
    
    double globalMaxError;
    MPI_Allreduce(&localMaxError, &globalMaxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    return globalMaxError;
}

int main(int argc, char** argv) {
    int procRank, numProcs;
    double startTime = 0, endTime = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    int sliceHeight = NZ / numProcs;
    
    double* currentValues = (double*)malloc(sizeof(double) * NX * NY * sliceHeight);
    double* nextValues = (double*)malloc(sizeof(double) * NX * NY * sliceHeight);
    double* topGhost = (double*)malloc(sizeof(double) * NX * NY);
    double* bottomGhost = (double*)malloc(sizeof(double) * NX * NY);
    
    double jacobiDenominator = 1.0 / (2.0/(DX*DX) + 2.0/(DY*DY) + 2.0/(DZ*DZ) + PARAM_A);
    
    if (procRank == 0) startTime = MPI_Wtime();
    
    for (int z = 0; z < sliceHeight; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                double xCoord = X_MIN + x * DX;
                double yCoord = Y_MIN + y * DY;
                double zCoord = Z_MIN + (z + sliceHeight * procRank) * DZ;
                
                if (x == 0 || x == NX-1 || y == 0 || y == NY-1) {
                    currentValues[IDX3D(x, y, z)] = exactSolution(xCoord, yCoord, zCoord);
                    nextValues[IDX3D(x, y, z)] = exactSolution(xCoord, yCoord, zCoord);
                } else {
                    currentValues[IDX3D(x, y, z)] = 0.0;
                    nextValues[IDX3D(x, y, z)] = 0.0;
                }
            }
        }
    }
    
    if (procRank == 0) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                double xCoord = X_MIN + x * DX;
                double yCoord = Y_MIN + y * DY;
                currentValues[IDX3D(x, y, 0)] = exactSolution(xCoord, yCoord, Z_MIN);
                nextValues[IDX3D(x, y, 0)] = exactSolution(xCoord, yCoord, Z_MIN);
            }
        }
    }
    
    if (procRank == numProcs - 1) {
        int z = sliceHeight - 1;
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                double xCoord = X_MIN + x * DX;
                double yCoord = Y_MIN + y * DY;
                currentValues[IDX3D(x, y, z)] = exactSolution(xCoord, yCoord, Z_MIN + Z_SIZE);
                nextValues[IDX3D(x, y, z)] = exactSolution(xCoord, yCoord, Z_MIN + Z_SIZE);
            }
        }
    }
    
    double* tempPtr;
    int iterationCount = 0;
    MPI_Request requests[4];
    
    char localConverged, globalConverged;
    do {
        localConverged = 1;
        
        tempPtr = currentValues;
        currentValues = nextValues;
        nextValues = tempPtr;
        
        if (procRank != 0) {
            MPI_Isend(currentValues, NX * NY, MPI_DOUBLE, procRank - 1, 10, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(bottomGhost, NX * NY, MPI_DOUBLE, procRank - 1, 20, MPI_COMM_WORLD, &requests[1]);
        }
        
        if (procRank != numProcs - 1) {
            MPI_Isend(&currentValues[IDX3D(0, 0, sliceHeight - 1)], NX * NY, MPI_DOUBLE, procRank + 1, 20, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(topGhost, NX * NY, MPI_DOUBLE, procRank + 1, 10, MPI_COMM_WORLD, &requests[3]);
        }
        
        updateInteriorPoints(procRank, sliceHeight, currentValues, nextValues, &localConverged, jacobiDenominator);
        
        if (procRank != 0) {
            MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
        }
        
        if (procRank != numProcs - 1) {
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        }
        
        updateBottomLayer(procRank, sliceHeight, currentValues, nextValues, bottomGhost, &localConverged, jacobiDenominator);
        updateTopLayer(procRank, sliceHeight, currentValues, nextValues, topGhost, &localConverged, numProcs, jacobiDenominator);
        
        MPI_Allreduce(&localConverged, &globalConverged, 1, MPI_CHAR, MPI_LAND, MPI_COMM_WORLD);
        
        if (procRank == 0) iterationCount++;
        
    } while (!globalConverged);
    
    if (procRank == 0) endTime = MPI_Wtime();
    
    double maxError = computeMaxError(procRank, sliceHeight, nextValues);
    
    if (procRank == 0) {
        printf("Number of iterations: %d\n", iterationCount);
        printf("Time: %lf sec.\n", (endTime - startTime));
        printf("Max difference: %.10lf\n", maxError);
    }
    
    free(currentValues);
    free(nextValues);
    free(topGhost);
    free(bottomGhost);
    
    MPI_Finalize();
    return 0;
}

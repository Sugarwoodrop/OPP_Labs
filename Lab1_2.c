#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 20000

double Norma(double* Vector) {
	double sum = 0;
	double final_number;
    #pragma omp for
	for (int i = 0; i < N; ++i) {
		#pragma omp atomic
		sum += Vector[i] * Vector[i] ;
	}
	final_number = sqrt(sum);

	return final_number;
}

void Multiplication(double* final_vector, double* Matrix, double* Vector) {
    #pragma omp for
	for (int i = 0; i < N; ++i) {
		double sum = 0;
		for (int j = 0; j < N; ++j) {
			sum += Matrix[i * N + j] * Vector[j];
		}
		final_vector[i] = sum;
	}
    
}

void Minus(double* result,double* first_vector, double* second_vector) {
    #pragma omp for
	for (int i = 0; i < N; ++i) {
		result[i] = first_vector[i] - second_vector[i];
	}
}

void Multiplication_Scalar(double* Vector) {
	#pragma omp for
	for (int i = 0; i < N; ++i) {
		Vector[i] = Vector[i] * 0.01;
	}
}


int main(void)
{
	omp_set_num_threads(16);
	double* matrix = (double*)calloc(N * N, sizeof(double));
	double* desired_vector = (double*)calloc(N, sizeof(double));
	double* arbitrary_vector = (double*)calloc(N, sizeof(double));
	if (!matrix) return 1;
	if (!desired_vector) return 1;
	if (!arbitrary_vector) return 1;

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if (i == j) { 
				matrix[i * N + j] = 2.0;
			}
			else {
				matrix[i * N + j] = 1.0;
			}
		}
		arbitrary_vector[i] = sin((2 * 3.14159 * i)/N);
	}

    double* vector = (double*)calloc(N, sizeof(double));
	double* mult = (double*)calloc(N, sizeof(double));
    double* final = (double*)calloc(N, sizeof(double));

    double start = omp_get_wtime();

	double norm_vector;
	double norm_final;
	int flag = 1;
#pragma omp parallel
{
	Multiplication(vector, matrix, arbitrary_vector);
	Multiplication(mult, matrix, desired_vector);
    Minus(final, mult, vector);

    norm_vector = Norma(vector);  
    norm_final = Norma(final); 

	while (flag) {
        Multiplication_Scalar(final);
                
        Minus(desired_vector,desired_vector, final);
		Multiplication(mult,matrix, desired_vector);
        Minus(final, mult, vector);
		norm_final = Norma(final);
		if(!((norm_final / norm_vector) > 1e-5)){
			flag = 0;	
		}
	}
}
	double end = omp_get_wtime();
	printf("%f\n", end - start);

	free(matrix);
	free(desired_vector);
	free(vector);
	free(mult);
	free(final);
	return 0;
}

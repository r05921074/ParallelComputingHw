#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define L1 1024
#define L2 1024
#define L3 1024
#define INIT_BOUND 30

int main(int argc, char *argv[]) {
	int i, j, k;
	int rank, nprocs;
	int dst_proc = 0, pass = 1;
	float *src_A = (float *)malloc(L1 * L2 * sizeof(float));
	float *src_B = (float *)malloc(L2 * L3 * sizeof(float));
	float *res = (float *)calloc(L1 * L3, sizeof(float));
	float *val = (float *)calloc(L1 * L3, sizeof(float));

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("rank %d in %d process\n", rank, nprocs);

	/* ========== Assign values to array A and B ========== */
	srand(time(NULL));
	for (i = 0; i < L1; i++) {
		for (j = 0; j < L2; j++) {
			src_A[i * L2 + j] = rand() % INIT_BOUND;
		}
	}
	for (i = 0; i < L2; i++) {
		for (j = 0; j < L3; j++) {
			src_B[i * L3 + j] = rand() % INIT_BOUND;
		}
	}

	/* ========== Transpose B matrix ========== */
	float tmp;
	float *src_B_T = (float *)malloc(L3 * L2 * sizeof(float));
	for (i = 0; i < L2; i++) {
		for (j = 0; j < L3; j++) {
			src_B_T[j * L3 + i] = src_B[i * L2 + j];
		}
	}

	/* ========== Calculate correct answers by destination processor ========== */
	struct timeval starttime, endtime;
	double executime;
	if (rank == dst_proc) {
		gettimeofday(&starttime, NULL);
		for (i = 0; i < L1; i++) {
			for (j = 0; j < L3; j++) {
				for (k = 0; k < L2; k++) {
					val[i * L3 + j] += src_A[i * L2 + k] * src_B_T[j * L3 + k];
				}
			}
		}
		gettimeofday(&endtime, NULL);
		executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
		executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
		printf("Sequential CPU time: %13lf msec\n", executime);
	}

	/* ========== Every processor calculates its own range ========== */
	int cal_rows = L1 / nprocs;
	int extra_rows = L1 % nprocs;  // if L1 is not a multiple of nb_processors
	int start_row = cal_rows * rank, end_row;
	gettimeofday(&starttime, NULL);
	if (rank == (nprocs - 1)) {
		end_row = cal_rows * (rank + 1) + extra_rows;
		for (i = start_row; i < end_row; i++) {
			for (j = 0; j < L3; j++) {
				for (k = 0; k < L2; k++) {
					res[i * L3 + j] += src_A[i * L2 + k] * src_B_T[j * L3 + k];
				}
			}
		}
	}
	else {
		end_row = cal_rows * (rank + 1);
		for (i = start_row; i < end_row; i++) {
			for (j = 0; j < L3; j++) {
				for (k = 0; k < L2; k++) {
					res[i * L3 + j] += src_A[i * L2 + k] * src_B_T[j * L3 + k];
				}
			}
		}
	}

	/* ========== Send and receive data ========== */
	int nb_data;
	if (rank == dst_proc) {
		for (i = 1; i < nprocs; i++) {
			if (i == (nprocs - 1)) {
				nb_data = (cal_rows + extra_rows) * L3;
				MPI_Recv(&res[i * cal_rows * L3], nb_data, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
			}
			else {
				nb_data = cal_rows * L3;
				MPI_Recv(&res[i * cal_rows * L3], nb_data, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
			}
		}
		gettimeofday(&endtime, NULL);
		executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
		executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
		printf("MPI calculates time: %13lf msec\n", executime);
	}
	else {
		if (rank == (nprocs - 1)) {
			nb_data = (cal_rows + extra_rows) * L3;
			MPI_Send(&res[start_row * L3], nb_data, MPI_FLOAT, dst_proc, 0, MPI_COMM_WORLD);
		}
		else {
			nb_data = cal_rows * L3;
			MPI_Send(&res[start_row * L3], nb_data, MPI_FLOAT, dst_proc, 0, MPI_COMM_WORLD);
		}
	}

	/* ========== Check if answers correct ========== */
	if (rank == 0) {
		for (i = 0; i < L1; i++) {
			for (j = 0; j < L3; j++) {
				if(val[i * L3 + j] != res[i * L3 + j]) {
					printf("val[%d][%d] = %2.0f, res[%d][%d] = %2.0f\n", i, j, val[i * L3 + j], i, j, res[i * L3 + j]);
					pass = 0;
				}
			}
		}
		printf("Test %s\n", (pass)?"PASSED":"FAILED");
	}

	MPI_Finalize();
	free(src_A);
	free(src_B);
	free(res);
	free(val);

	return 0;
}
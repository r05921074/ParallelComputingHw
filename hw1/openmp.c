#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define ARRAY_LENGTH 1024
#define TEST_TIMES 10

int main(int argc, int *argv[])
{
	//Initialize variables
	int i, j, k;
	int **A, **B, **CC, **Answers;
	A = (int **)calloc(ARRAY_LENGTH, sizeof(int *));
	B = (int **)calloc(ARRAY_LENGTH, sizeof(int *));
	CC = (int **)calloc(ARRAY_LENGTH, sizeof(int *));
	Answers = (int **)calloc(ARRAY_LENGTH, sizeof(int *));
	struct timespec t_start, t_end;
	double elapsedTime;
	for (i = 0; i < ARRAY_LENGTH; i++)
	{
		A[i] = (int *)calloc(ARRAY_LENGTH, sizeof(int));
		B[i] = (int *)calloc(ARRAY_LENGTH, sizeof(int));
		CC[i] = (int *)calloc(ARRAY_LENGTH, sizeof(int)); 
		Answers[i] = (int *)calloc(ARRAY_LENGTH, sizeof(int)); 
	}

	//Assign values to two matrix
	for (i = 0; i < ARRAY_LENGTH; i++)
	{
		for (j = 0; j < ARRAY_LENGTH; j++)
		{
			A[i][j] = i + j;
			B[i][j] = i * j;
		}
	}

	//Calculate correct answers
	for (i = 0; i < ARRAY_LENGTH; i++)
	{
		for (j = 0; j < ARRAY_LENGTH; j++)
		{
			Answers[i][j] = 0;
			for (k = 0; k < ARRAY_LENGTH; k++)
				Answers[i][j] += A[i][k] * B[k][j];
		}
	}

	//Main loop
	int flag, run_count;
	double accumulated_run_time = 0.0;
	for (run_count = 0; run_count < TEST_TIMES; run_count++)
	{
		//Calculate matrix multiplication and measure elasped time
		clock_gettime(CLOCK_REALTIME, &t_start);
		#pragma omp parallel for private(i, j, k)
		for (i = 0; i < ARRAY_LENGTH; i++)
		{
			for (j = 0; j < ARRAY_LENGTH; j++)
			{
				CC[i][j] = 0;
				for (k = 0; k < ARRAY_LENGTH; k++)
					CC[i][j] += A[i][k] * B[k][j];
			}
		}
		clock_gettime(CLOCK_REALTIME, &t_end);

		//Check if the results are wrong
		flag = 0;
		for (i = 0; i < ARRAY_LENGTH && flag != 1; i++)
		{
			for (j = 0; j < ARRAY_LENGTH; j++)
			{
				if (CC[i][j] != Answers[i][j])
				{
					flag = 1;
					printf("Doesn't match the answers!!n");
				}
			}
		}
		if (flag == 1)
			break;

		elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
		elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
		accumulated_run_time += elapsedTime;
		printf("Iteration: %d, elapsedTime: %lf ms\n", run_count + 1, elapsedTime);
	}
	printf("Average run time: %lf ms\n", accumulated_run_time / TEST_TIMES);

	//free memory
	for (i = 0; i < ARRAY_LENGTH; i++)
	{
		free(A[i]);
		free(B[i]);
		free(CC[i]);
	}
	free(A);
	free(B);
	free(CC);
	return 0;
}
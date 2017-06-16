#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
//there are 4 threads in mac

//#define DEBUG
//#define CHECK
//#define WRITEFILE

#define THREADS_NUM 4
#define DATA_NUM 2500000
#define NUM_BOUND 2147483647

int compare(const void *arg1, const void *arg2) {
	int ret = *(int*)(arg1) - *(int*)(arg2);
	if (ret > 0) return 1;
	if (ret < 0) return -1;
	return 0;
}

int findDesThread(int split_bound[], int element) {
	if (0 < element && element <= split_bound[0]) return 0;
	else if (split_bound[0] < element && element <= split_bound[1]) return 1;
	else if (split_bound[1] < element && element <= split_bound[2]) return 2;
	else return 3;
}

int main(void) {
	int i, j;
	/* ========== Initialize data structures ========== */
	int **pRawNums = (int **)calloc(THREADS_NUM, sizeof(int *));
	int **pDataSorted = (int **)calloc(THREADS_NUM, sizeof(int *));
	for (int i = 0; i < THREADS_NUM; i++) {
		pRawNums[i] = (int *)calloc(DATA_NUM, sizeof(int));
		pDataSorted[i] = (int *)calloc(2 * DATA_NUM, sizeof(int));
	}
	int *pAnswer = (int *)calloc(THREADS_NUM * DATA_NUM, sizeof(int));
	int *pCorrect = (int *)calloc(THREADS_NUM * DATA_NUM, sizeof(int));

	/* ========== Initialize parameters ========== */
	int candidates[THREADS_NUM * (THREADS_NUM-1)] = {0};
	int split_bound[THREADS_NUM - 1] = {0};
	int thread_element_count[THREADS_NUM] = {0};
	int s_index[THREADS_NUM+1] = {0};
	omp_set_num_threads(THREADS_NUM);

	/* ========== Initialize variables ========== */
	int c_num = THREADS_NUM * (THREADS_NUM-1);  //candidates number
	int c_index = 0;
	int c_Count = THREADS_NUM - 1;  //each thread contain how many candidates
	int c_Interval = (DATA_NUM / (THREADS_NUM - 1));  //candidate_Interval

	srand(time(NULL));
	struct timespec t_start, t_end;
	double elapsedTime, totalTime = 0;

	/* ========== Assign numbers to arrays ========== */
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < DATA_NUM; j++) {
			pRawNums[i][j] = (rand() % NUM_BOUND) + 1;
			//printf("%4d", pRawNums[i][j]);
			pCorrect[i * DATA_NUM + j] = pRawNums[i][j];
		}
		//printf("\n");
	}

#ifdef WRITEFILE
	FILE *OriData = fopen("OriData.csv", "w");
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < DATA_NUM; j++) {
			fprintf(OriData, "%d,", pRawNums[i][j]);
		}
		fprintf(OriData, "\n");
	}
	fclose(OriData);
#endif

	/* ========== Choose (THREADS_NUM - 1) candidates from each chunk randomly ========== */
	clock_gettime( CLOCK_REALTIME, &t_start);
	int count = 0;
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < THREADS_NUM-1; j++) {
			//candidates[count] = pRawNums[i][(j+1) * (DATA_NUM / (THREADS_NUM)) - 1];
			int e_index = rand() % DATA_NUM;
			candidates[count] = pRawNums[i][e_index];
			count++;
		}
	}

#ifdef DEBUG
	printf("Candidates not sorted\n");
	for (i = 0; i < (THREADS_NUM * (THREADS_NUM-1)); i++) {
		printf("%d ", candidates[i]);
	}
	printf("\n");
#endif

#ifdef WRITEFILE
	FILE *Candidates_NS = fopen("Candidates_NS.csv", "w");
	for (i = 0; i < (THREADS_NUM * (THREADS_NUM-1)); i++) {
		fprintf(Candidates_NS, "%d,", candidates[i]);
	}
	fclose(Candidates_NS);
#endif

	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%-40s %lf ms\n", "Choose candidates:", elapsedTime);
	totalTime += elapsedTime;

	/* ========== Sort candidates and choose (THREADS_NUM - 1) real splitters ========== */
	clock_gettime(CLOCK_REALTIME, &t_start);
	qsort(candidates, (THREADS_NUM * (THREADS_NUM-1)), sizeof(int), compare);

#ifdef DEBUG
	printf("Candidates sorted\n");
	for (i = 0; i < (THREADS_NUM * (THREADS_NUM-1)); i++) {
		printf("%d ", candidates[i]);
	}
	printf("\n");
#endif

#ifdef WRITEFILE
	FILE *Candidates_S = fopen("Candidates_S.csv", "w");
	for (i = 0; i < (THREADS_NUM * (THREADS_NUM-1)); i++) {
		fprintf(Candidates_S, "%d,", candidates[i]);
	}
	fclose(Candidates_S);
#endif

	for (i = 0; i < (THREADS_NUM - 1); i++) {
		//split_bound[i] = candidates[i * THREADS_NUM + (THREADS_NUM - 1) - 2];
		split_bound[i] = candidates[(int)(THREADS_NUM * (i + 0.5))];
		//printf("%d ", (int)(THREADS_NUM * (i + 0.5)));
	}
	//printf("\n");

#ifdef DEBUG
	printf("Real splitters\n");
	for (i = 0; i < (THREADS_NUM - 1); i++) {
		printf("%d ", split_bound[i]);
	}
	printf("\n");
#endif

#ifdef WRITEFILE
	FILE *R_splitter = fopen("Real_splitter.csv", "w");
	for (i = 0; i < (THREADS_NUM - 1); i++) {
		fprintf(R_splitter, "%d,", split_bound[i]);
	}
	fclose(R_splitter);
#endif

	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%-40s %lf ms\n", "Sort candidates and choose splitters:", elapsedTime);
	totalTime += elapsedTime;

	/* ========== Use splitters to seperate data with different size ========== */
	clock_gettime(CLOCK_REALTIME, &t_start);
	int desThread = 0;
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < DATA_NUM; j++) {
			desThread = findDesThread(split_bound, pRawNums[i][j]);
			pDataSorted[desThread][thread_element_count[desThread]] = pRawNums[i][j];
			thread_element_count[desThread]++;
		}	
	}

	for (i = 0; i < THREADS_NUM; i++)
		printf("%d ", thread_element_count[i]);
	printf("\n");

#ifdef DEBUG
	printf("After dispatch datas by splitters\n");
	for (i = 0; i < THREADS_NUM; i++) {
		printf("%d ******************************************************\n", i);
		for (j = 0; j < thread_element_count[i]; j++) {
			printf("%d, ", pDataSorted[i][j]);
		}
		printf("\n");
	}
#endif

#ifdef WRITEFILE
	FILE *Bucket_NS = fopen("Bucket_NS.csv", "w");
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < thread_element_count[i]; j++) {
			fprintf(Bucket_NS, "%d,", pDataSorted[i][j]);
		}
		fprintf(Bucket_NS, "\n");
	}
	fclose(Bucket_NS);
#endif

	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%-40s %lf ms\n", "Seperate data into chunks by values:", elapsedTime);
	totalTime += elapsedTime;

	/* ========== Calculate new chunks' start indices ========== */
	clock_gettime(CLOCK_REALTIME, &t_start);
	for (i = 1; i < THREADS_NUM; i++)
		s_index[i] = s_index[i-1] + thread_element_count[i-1];
	s_index[THREADS_NUM] = THREADS_NUM * DATA_NUM;

#ifdef DEBUG
	printf("Start Positions\n");
	for (i = 0; i < THREADS_NUM + 1; i++)
		printf("%d ", s_index[i]);
	printf("\n");
#endif

	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%-40s %lf ms\n", "Calculate start indices:", elapsedTime);
	totalTime += elapsedTime;

	/* ========== Use quick sort to sort each chunk ========== */
	clock_gettime(CLOCK_REALTIME, &t_start);
	#pragma omp parallel for private(i)
	for (i = 0; i < THREADS_NUM; i++)
		qsort(pDataSorted[i], thread_element_count[i], sizeof(int), compare);

#ifdef DEBUG
	printf("After last chunk sorted\n");
	for (i = 0; i < THREADS_NUM; i++) {
		printf("%d ******************************************************\n", i);
		for (j = 0; j < thread_element_count[i]; j++) {
			printf("%d, ", pDataSorted[i][j]);
		}
		printf("\n");
	}
#endif

#ifdef WRITEFILE
	FILE *ChunkSorted = fopen("ChunkSorted.csv", "w");
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < thread_element_count[i]; j++) {
			fprintf(ChunkSorted, "%d,", pDataSorted[i][j]);
		}
		fprintf(ChunkSorted, "\n");
	}
	fclose(ChunkSorted);
#endif

	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%-40s %lf ms\n", "Apply quick sort to each chunk:", elapsedTime);
	totalTime += elapsedTime;

	/* ========== Merge previous result into a new array ========== */
	clock_gettime(CLOCK_REALTIME, &t_start);
	#pragma omp parallel for private(i, j)
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < thread_element_count[i]; j++)
			pAnswer[s_index[i] + j] = pDataSorted[i][j];
	}

#ifdef WRITEFILE
	FILE *Parallel_Ans = fopen("Parallel_Ans.csv", "w");
	for (i = 0; i < THREADS_NUM * DATA_NUM; i++)
		fprintf(Parallel_Ans, "%d,", pAnswer[i]);
	fclose(Parallel_Ans);
#endif

	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%-40s %lf ms\n", "Merge data:", elapsedTime);
	totalTime += elapsedTime;

	printf("\nParallel sample sort(static) elapsedTime: %lf ms\n", totalTime);

	/* ========== Calculating correct answers for testing ========== */
	clock_gettime(CLOCK_REALTIME, &t_start);
	qsort(pCorrect, THREADS_NUM * DATA_NUM, sizeof(int), compare);
	clock_gettime( CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("Plugin quick sort elapsedTime: %lf ms\n", elapsedTime);

#ifdef CHECK
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < DATA_NUM; j++) {
			printf("%d, %d\n", i, j);
			printf("%d---> Real Answer: %d, pAnswer: %d    ", i, pCorrect[i * DATA_NUM + j], pAnswer[i * DATA_NUM + j]);
			if (pCorrect[i * DATA_NUM + j] == pAnswer[i * DATA_NUM + j]) printf("O\n");
			else printf("X\n");
		}
	}
#endif

	/* ========== Check if answers correct ========== */
	int tmpcount = 0;
	for (i = 0; i < THREADS_NUM; i++) {
		for (j = 0; j < DATA_NUM; j++) {
			if (pCorrect[i * DATA_NUM + j] != pAnswer[tmpcount]) {
				printf("Answer is wrong, index is at %d\n", i * DATA_NUM + j);
				break;
			}
			tmpcount++;
		}
	}

	/* ========== Release memory ========== */
	for (int i = 0; i < THREADS_NUM; i++) {
		free(pRawNums[i]);
		free(pDataSorted[i]);
	}
	free(pRawNums);
	free(pCorrect);
	free(pDataSorted);
	free(pAnswer);

	return 0;
}
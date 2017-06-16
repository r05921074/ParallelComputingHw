#include <iostream>
#include <vector>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define ITERATION 1500
#define DIM 100
#define SIZE 60  /* Has its limitation due to the shared memory size on GPU */
#define UPPER 100
#define LOWER -100
#define S_UPPER (UPPER - LOWER) / 10
#define S_LOWER (LOWER - UPPER) / 10


#define PI 3.1415926535897932384626433832795
#define W 0.8
#define C1 2.8
#define C2 1.3

#define UNCHANGE_THRES 20

//#define MAX
#define MIN

//#define DEBUG

#define FILE_PATH "CUDA_PSO_F6_DIM100_SIZE60_ITER1500.csv"

#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
	#define printf(f, ...) ((void)(f, __VA_ARGS__), 0)
#endif


struct local_best {
	double fitness_value;
	double dimension[DIM];
	double speed[DIM];
};

struct particle {
	int unchange_time;
	double fitness_value;
	double dimension[DIM];
	double speed[DIM];
	local_best pbest;
};

struct global_best {
	double fitness_value;
	double dimension[DIM];
};

struct Node {
	double best;
	int idx;
};

void initialize(particle h_particles[SIZE]);
void find_gbest(particle h_particles[SIZE], global_best *h_gbest);
double fitness(particle p);
double formula1(particle p);
double formula2(particle p);
double formula3(particle p);
double formula4(particle p);
double formula5(particle p);
double formula6(particle p);
void TransportData(particle h_particles[SIZE], global_best *h_gbest);

__global__ void RunKernel(particle *d_particles, global_best *d_gbest, int *d_seed);
__device__ void FetchGlobal(Node *paras, double dimension[DIM], double valBestFitness[SIZE], double valBestParas[SIZE][DIM], int nb_iter);
__device__ void Move(Node *paras, particle *d_particles, double dimension[DIM], curandState_t *state, int nb_iter);
__device__ void Cal_Fitness(particle *d_particles, curandState_t *state, int nb_iter);
__device__ void UploadLocal(particle *d_particles, double valBestFitness[SIZE], double valBestParas[SIZE][DIM], int nb_iter);
__device__ double Fitness(particle p);
__device__ double Formula1(particle p);
__device__ double Formula2(particle p);
__device__ double Formula3(particle p);
__device__ double Formula4(particle p);
__device__ double Formula5(particle p);
__device__ double Formula6(particle p);
__device__ void GetFinalParas(global_best *gbest, double valBestFitness[SIZE], double valBestParas[SIZE][DIM]);

__device__ void Print_Best_Fitness(double valBestFitness[SIZE], int nb_iter);
__device__ void Print_Best_Paras(double valBestParas[SIZE][DIM], int nb_iter);
__device__ void PrintParticleBest(particle p);
__device__ void PrintLocalFitness(particle p);
__device__ void PrintFormula(particle p);
__device__ void Print_C1(double C1_term);
__device__ void Print_C2(double C2_term);
__device__ void Print_NL(void);
__device__ void DEBUG_BEST(particle *d_particles);
__device__ void Print_Speed(particle p);
__device__ void Print_Dimension(particle p);

/***************************/
/****** Host functions *****/
/***************************/
int main() {
	particle h_particles[SIZE];
	global_best h_gbest;
	initialize(h_particles);
	find_gbest(h_particles, &h_gbest);
	TransportData(h_particles, &h_gbest);

	return 0;
}

void initialize(particle h_particles[SIZE]) {
	srand(time(NULL));
	for(int i = 0; i < SIZE; i++) {
		for(int j = 0; j < DIM; j++) {
			double temp = (double)rand() / (RAND_MAX + 1.0);
			h_particles[i].dimension[j] = temp * (UPPER - LOWER) + LOWER;
			h_particles[i].pbest.dimension[j] = h_particles[i].dimension[j];

			temp = (double)rand() / (RAND_MAX + 1.0);
			h_particles[i].speed[j] = temp * (UPPER - LOWER) + LOWER;
		}
		h_particles[i].fitness_value = fitness(h_particles[i]);
		h_particles[i].pbest.fitness_value = h_particles[i].fitness_value;
	}
}

void find_gbest(particle h_particles[SIZE], global_best *h_gbest) {
	for(int i = 0; i < SIZE; i++) {
		#ifdef MAX 
		if(h_particles[i].pbest.fitness_value > h_gbest->fitness_value) {
			for(int j = 0; j < DIM; j++) {
				h_gbest->dimension[j] = h_particles[i].pbest.dimension[j]; 
			}
			h_gbest->fitness_value = h_particles[i].pbest.fitness_value;
		}
		#endif

		#ifdef MIN
		if(h_particles[i].pbest.fitness_value < h_gbest->fitness_value) {
			for(int j = 0; j < DIM; j++) {
				h_gbest->dimension[j] = h_particles[i].pbest.dimension[j]; 
			}
			h_gbest->fitness_value = h_particles[i].pbest.fitness_value;
		}
		#endif
	}
}

double fitness(particle p) {
	double sum;
	//sum = formula1(p);
	// sum = formula2(p);
	// sum = formula3(p);
	// sum = formula4(p);
	// sum = formula5(p);
	sum = formula6(p);

	return sum;
}

double formula1(particle p) {
	double sum = 0;
	for(int i = 0; i < DIM; i++) {
		sum += sin(p.dimension[i]) / p.dimension[i];
	}
	return sum;
}

double formula2(particle p) {
	double sum = 0;
	for(int i = 0; i < DIM; i++)
	{
		sum += p.dimension[i] * p.dimension[i];
	}
	return sum;
}

double formula3(particle p) { 
	double sum = 20 + pow(p.dimension[0], 2) + pow(p.dimension[1], 2) - 10 * (cos(2 * PI * p.dimension[0]) + cos(2 * PI * p.dimension[1]));
	return sum;
}

double formula4(particle p) {
	double sum = 0.7 + pow(p.dimension[0], 2) + pow(p.dimension[1], 2) - 0.3 * cos(3 * PI * p.dimension[0]) - 0.4 * cos(4 * PI * p.dimension[1]);
	return sum;
}

double formula5(particle p) {
	double sum = pow(p.dimension[0], 2) + pow(p.dimension[1], 2) + pow(p.dimension[2], 2) + pow(p.dimension[3], 2) + pow(p.dimension[4], 2);
	return sum;
}

double formula6(particle p) {
	double sum = 0;
	for (int i = 0; i < DIM; i++)
		sum += pow(p.dimension[i], 3);
	return fabs(sum);
}

void TransportData(particle h_particles[SIZE], global_best *h_gbest) {
	size_t size_particles = SIZE * sizeof(particle);
	size_t size_gbest = 1 * sizeof(global_best);
	size_t size_seeds = SIZE * sizeof(int);
	particle *d_particles;
	global_best *d_gbest, res_gbest;

	int *h_seeds = (int *)malloc(SIZE * sizeof(int));
	for (int i = 0; i < SIZE; i++) {
		h_seeds[i] = (int)rand() % 1000;
	}
	int *d_seeds;

	/* ========== Allocate memory on GPU ========== */

	cudaMalloc((void **)&d_particles, size_particles);
	cudaMemcpy(d_particles, h_particles, size_particles, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_gbest, size_gbest);
	cudaMemcpy(d_gbest, h_gbest, size_gbest, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&res_gbest, size_gbest);

	cudaMalloc((void **)&d_seeds, size_seeds);
	cudaMemcpy(d_seeds, h_seeds, size_seeds, cudaMemcpyHostToDevice);

	/* ========== Get start time event ========== */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/* ========== Check if particles less than max_threads in one block ========== */
	if (SIZE <= 1024) 
		RunKernel<<<1, SIZE>>>(d_particles, d_gbest, d_seeds);

	/* ========== Get stop time event ========== */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	/* ========== Compute execution time ========== */
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU time: %13f msec\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* ========== Copy result back from GPU ========== */
	cudaMemcpy(&res_gbest, d_gbest, size_gbest, cudaMemcpyDeviceToHost);

	/* ========== Print results on screen ========== */
	printf("Global best: %lf\n", res_gbest.fitness_value);

	/* ========== Free memory ========== */
	cudaFree(d_particles);
	cudaFree(d_gbest);

	/* ========== Write output file ========== */
	FILE *f = fopen(FILE_PATH, "a");
	fprintf(f, "%.15lf,%lf\n", res_gbest.fitness_value, elapsedTime);
	fclose(f);
}

/***************************/
/***** Device functions ****/
/***************************/
__global__ void RunKernel(particle *d_particles, global_best *d_gbest, int *d_seeds) {
	int particle_id = threadIdx.x;
	curandState_t state;  /* A parameter which curand needed */
	curand_init(d_seeds[particle_id], particle_id, 0, &state);  /* Random seed for cuda random APIs */

	Node paras;  /* A temporary data structure to save global best and index at the moment */
	double dimension[DIM] = { 0.0 };  /* Used to save best parameters from other thread */

	__shared__ double valBestFitness[SIZE];  /* Each thread upload its local best fitness into this array, index is its thread id */
	__shared__ double valBestParas[SIZE][DIM];  /* Each thread upload its local parameters which can get lobal best into this array */

	/* ========== Initialize local best fitness to shared memory with default value ========== */
	#ifdef MAX
	valBestFitness[particle_id] = DBL_MIN;
	#endif
	#ifdef MIN
	valBestFitness[particle_id] = DBL_MAX;
	#endif
	__syncthreads();
	UploadLocal(d_particles, valBestFitness, valBestParas, -1);

	/* ========== The main body of PSO algorithm ========== */
	for (int i = 0; i < ITERATION; i++) {
		/* ========== Fetch the global best fitness and parameters at the moment ========== */
		FetchGlobal(&paras, dimension, valBestFitness, valBestParas, i);

		/* ========== Update the particle position and speed according to the parameters fetched ========== */
		Move(&paras, d_particles, dimension, &state, i);

		/* ========== Recalculate the fitness ========== */
		Cal_Fitness(d_particles, &state, i);

		/* ========== Recalculate the fitness ========== */
		UploadLocal(d_particles, valBestFitness, valBestParas, i);

		/* ========== DEBUG SECTION ========== */

	}
	__syncthreads();

	/* ========== Write data to return structure ========== */
	GetFinalParas(d_gbest, valBestFitness, valBestParas);
}

__device__ void FetchGlobal(Node *paras, double dimension[DIM], double valBestFitness[SIZE], double valBestParas[SIZE][DIM], int nb_iter) {
	/* ========== Each thread update its own saved best paras(paras) at the time it checks shared memories ========= */
	#ifdef MAX
	paras->best = DBL_MIN;
	for (int i = 0; i < SIZE; i++) {
		if (valBestFitness[i] > paras->best) {
			paras->best = valBestFitness[i];
			paras->idx = i;
			for (int j = 0; j < DIM; j++)
				dimension[j] = valBestParas[i][j];
		}
	}
	#endif

	#ifdef MIN
	paras->best = DBL_MAX;
	for (int i = 0; i < SIZE; i++) {
		if (valBestFitness[i] < paras->best) {
			paras->best = valBestFitness[i];
			paras->idx = i;
			for (int j = 0; j < DIM; j++)
				dimension[j] = valBestParas[i][j];
		}
	}
	#endif
}

__device__ void Move(Node *paras, particle *d_particles, double dimension[DIM], curandState_t *state, int nb_iter) {
	/* ========== Each thread update its own positions and speed due to the previous information it gets ========= */
	int particle_id = threadIdx.x;
	double C1_term, C2_term;
	double C1_rand, C2_rand;

	for (int i = 0; i < DIM; i++) {
		C1_rand = (double)curand(state);
		C2_rand = (double)curand(state);
		C1_term = C1 * C1_rand / (UINT_MAX + 1.0) * (d_particles[particle_id].pbest.dimension[i] - d_particles[particle_id].dimension[i]);
		C2_term = C2 * C2_rand / (UINT_MAX + 1.0) * (dimension[i] - d_particles[particle_id].dimension[i]);
		// printf("%d, %d, %lf, %lf\n", i, threadIdx.x, C1_rand, C2_rand);

		d_particles[particle_id].speed[i] = W * d_particles[particle_id].speed[i] + C1_term + C2_term;
		if (d_particles[particle_id].speed[i] > S_UPPER)
			d_particles[particle_id].speed[i] = S_UPPER;
		if (d_particles[particle_id].speed[i] < S_LOWER)
			d_particles[particle_id].speed[i] = S_LOWER;

		d_particles[particle_id].dimension[i] += d_particles[particle_id].speed[i];

		if (d_particles[particle_id].dimension[i] > UPPER) {
			double temp = (double)curand(state) / (UINT_MAX + 1.0);
			d_particles[particle_id].dimension[i] = temp * (UPPER - LOWER) + LOWER;
		}
		if (d_particles[particle_id].dimension[i] < LOWER) {
			double temp = (double)curand(state) / (UINT_MAX + 1.0);
			d_particles[particle_id].dimension[i] = temp * (UPPER - LOWER) + LOWER;
		}
	}
}

__device__ void Cal_Fitness(particle *d_particles, curandState_t *state, int nb_iter) {
	/* ========== Each thread (particle) calculates its fitness and check if it is greater than local best ========= */
	int particle_id = threadIdx.x;

	d_particles[particle_id].fitness_value = Fitness(d_particles[particle_id]);

	#ifdef MAX
	if (d_particles[particle_id].fitness_value > d_particles[particle_id].pbest.fitness_value) {
		for (int i = 0; i < DIM; i++) {
			d_particles[particle_id].pbest.dimension[i] = d_particles[particle_id].dimension[i];
		}
		d_particles[particle_id].pbest.fitness_value = d_particles[particle_id].fitness_value;
	}
	#endif

	#ifdef MIN
	if (d_particles[particle_id].fitness_value < d_particles[particle_id].pbest.fitness_value) {
		for (int i = 0; i < DIM; i++) {
			d_particles[particle_id].pbest.dimension[i] = d_particles[particle_id].dimension[i];
		}
		d_particles[particle_id].pbest.fitness_value = d_particles[particle_id].fitness_value;
	}
	#endif
}

__device__ void UploadLocal(particle *d_particles, double valBestFitness[SIZE], double valBestParas[SIZE][DIM], int nb_iter) {
	/* ========== Each thread uploads its local best to the shared memory, so other threads can access the results ========= */
	int particle_id = threadIdx.x;

	#ifdef MAX
	if (d_particles[particle_id].pbest.fitness_value > valBestFitness[particle_id]) {
		valBestFitness[particle_id] = d_particles[particle_id].pbest.fitness_value;
		for (int i = 0; i < DIM; i++) {
			valBestParas[particle_id][i] = d_particles[particle_id].pbest.dimension[i];
		}
	}
	#endif

	#ifdef MIN
	if (d_particles[particle_id].pbest.fitness_value < valBestFitness[particle_id]) {
		valBestFitness[particle_id] = d_particles[particle_id].pbest.fitness_value;
		for (int i = 0; i < DIM; i++) {
			valBestParas[particle_id][i] = d_particles[particle_id].pbest.dimension[i];
		}
	}
	#endif
}

__device__ double Fitness(particle p) {
	double sum;
	// sum = Formula1(p);
	// sum = Formula2(p);
	// sum = Formula3(p);
	// sum = Formula4(p);
	// sum = Formula5(p);
	sum = Formula6(p);

	return sum;
}

__device__ double Formula1(particle p) {
	double sum = 0;
	for(int i = 0; i < DIM; i++) {
		sum += sin(p.dimension[i]) / p.dimension[i];
	}
	return sum;
}

__device__ double Formula2(particle p) {
	double sum = 0;
	for(int i = 0; i < DIM; i++) {
		sum += p.dimension[i] * p.dimension[i];
	}
	return sum;
}

__device__ double Formula3(particle p) {
	double sum = 20 + pow(p.dimension[0], (double)2.0) + pow(p.dimension[1], (double)2.0) - 10 * (cos(2 * PI * p.dimension[0]) + cos(2 * PI * p.dimension[1]));
	return sum;
}

__device__ double Formula4(particle p) {
	double sum = 0.7 + pow(p.dimension[0], 2) + pow(p.dimension[1], 2) - 0.3 * cos(3 * PI * p.dimension[0]) - 0.4 * cos(4 * PI * p.dimension[1]);
	return sum;
}

__device__ double Formula5(particle p) {
	double sum = pow(p.dimension[0], (double)2.0) + pow(p.dimension[1], (double)2.0) + pow(p.dimension[2], (double)2.0) + pow(p.dimension[3], (double)2.0) + pow(p.dimension[4], (double)2.0);
	return sum;
}

__device__ double Formula6(particle p) {
	double sum = 0;
	for (int i = 0; i < DIM; i++)
		sum += pow(p.dimension[i], 3);
	return fabs(sum);
}

__device__ void GetFinalParas(global_best *gbest, double valBestFitness[SIZE], double valBestParas[SIZE][DIM]) {
	/* ========== After iterations finish, choose the best result and write it into the structure returned to host ========= */
	#ifdef MAX
	gbest->fitness_value = DBL_MIN;
	for (int i = 0; i < SIZE; i++) {
		if (valBestFitness[i] > gbest->fitness_value) {
			gbest->fitness_value = valBestFitness[i];
			for (int j = 0; j < DIM; j++)
				gbest->dimension[j] = valBestParas[i][j];
		}
	}
	#endif

	#ifdef MIN
	gbest->fitness_value = DBL_MAX;
	for (int i = 0; i < SIZE; i++) {
		if (valBestFitness[i] < gbest->fitness_value) {
			gbest->fitness_value = valBestFitness[i];
			for (int j = 0; j < DIM; j++)
				gbest->dimension[j] = valBestParas[i][j];
		}
	}
	#endif
}

__device__ void Print_Best_Fitness(double valBestFitness[SIZE], int nb_iter) {
	if (threadIdx.x == 0) {
		printf("\nIteration: %4d best fitness\n", nb_iter);
		for (int i = 0; i < SIZE; i++)
			printf("%11.6lf ", valBestFitness[i]);
		printf("\n");
	}
}

__device__ void Print_Best_Paras(double valBestParas[SIZE][DIM], int nb_iter) {
	if (threadIdx.x == 0) {
		printf("\nIteration: %4d best parameters\n", nb_iter);
		for (int i = 0; i < SIZE; i++) {
			printf("%d\n\t", i);
			for (int j = 0; j < DIM; j++)
				printf("%11.6lf ", valBestParas[i][j]);
			printf("\n");
		}
	}
}

// __device__ void PrintParticleBest(particle p) {
// 	printf("%d: %lf\n", threadIdx.x, p.pbest.fitness_value);
// }

// __device__ void PrintLocalFitness(particle p) {
// 	printf("%d: %lf\n", threadIdx.x, p.fitness_value);
// }

// __device__ void PrintFormula(particle p) {
// 	if (threadIdx.x == 0) {
// 		for (int i = 0; i < DIM; i++)
// 			printf("%lf, ", p.dimension[i]);
// 		printf("\n");
// 	}
// }

// __device__ void Print_C1(double C1_term) {
// 	if (threadIdx.x == 0)
// 		printf("%lf, ", C1_term);
// }

// __device__ void Print_C2(double C2_term) {
// 	if (threadIdx.x == 0)
// 		printf("%lf\n\n", C2_term);
// }

// __device__ void Print_NL(void) {
// 	if (threadIdx.x == 0)
// 		printf("\n");
// }

// __device__ void DEBUG_BEST(particle *d_particles) {
// 	for (int i = 0; i < SIZE; i++) {
// 		printf("%d: %lf, ", threadIdx.x, d_particles[threadIdx.x].pbest.fitness_value);
// 		for (int j = 0; j < SIZE; j++) {
// 			printf("%lf, ", d_particles[threadIdx.x].pbest.dimension[j]);
// 		}
// 		printf("\n");
// 	}
// }

// __device__ void Print_Speed(particle p) {
// 	for (int i = 0; i < DIM; i++) {
// 		printf("%lf, ", p.speed[i]);
// 	}
// 	printf("\n");
// }

// __device__ void Print_Dimension(particle p) {
// 	for (int i = 0; i < DIM; i++) {
// 		printf("%lf, ", p.dimension[i]);
// 	}
// 	printf("\n");
// }
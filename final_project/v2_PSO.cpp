#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>

using namespace std;

// #define DEBUG

#define ITERATION 1500
#define DIM 100
#define SIZE 350
#define UPPER 100
#define LOWER -100
#define S_UPPER (UPPER-LOWER)/10
#define S_LOWER (LOWER-UPPER)/10

#define PI 3.1415926535897932384626433832795
#define W 0.8
#define C1 2.8
#define C2 1.3

//#define MAX
#define MIN

#define FILE_PATH "C_PSO_F6_DIM100_SIZE350_ITER1500.csv"

struct local_best {
	double fitness_value;
	vector<double> dimension;
	vector<double> speed;
	local_best(): dimension(DIM),speed(DIM){};
};

struct particle {
	double fitness_value;
	vector<double> dimension;
	vector<double> speed;
	local_best pbest;
	particle(): dimension(DIM),speed(DIM){};
};

struct global_best {
	double fitness_value;
	vector<double> dimension;
	global_best(): fitness_value(INT8_MAX),dimension(DIM){};
};

double formula1(particle p) {
	double sum = 0;
	for(int i = 0;i < DIM;i++) {
		sum += sin(p.dimension[i])/p.dimension[i];
	}

	return sum;
}

double formula2(particle p) {
	double sum = 0;
	for(int i = 0;i < DIM;i++) {
		sum += p.dimension[i]*p.dimension[i];
	}

	return sum;
}

double formula3(particle  p) { 
	double sum = 20 + pow(p.dimension[0],2) + pow(p.dimension[1],2) - 10 * (cos(2 * PI * p.dimension[0]) + cos(2 * PI * p.dimension[1]));
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

void initialize(vector<particle>& p) {
	srand(time(NULL));
	for(int i = 0; i < SIZE; i++) {
		for(int j = 0; j < DIM; j++) {
			double temp = (double)rand()/(RAND_MAX + 1.0);
			p[i].dimension[j] = temp * (UPPER-LOWER)+LOWER;
			p[i].pbest.dimension[j] = p[i].dimension[j];

			temp = (double)rand()/(RAND_MAX + 1.0);
			p[i].speed[j] = temp * (UPPER-LOWER)+LOWER;
		}
		p[i].fitness_value = fitness(p[i]);
		p[i].pbest.fitness_value = p[i].fitness_value;
	}

	return;
}
 
void find_gbest(vector<particle> p,global_best& gbest) {
	for(int i = 0; i < SIZE; i++) {
	#ifdef MAX 
		if(p[i].pbest.fitness_value > gbest.fitness_value) {
			for(int j = 0; j < DIM; j++) {
				gbest.dimension[j] = p[i].pbest.dimension[j]; 
			}
			gbest.fitness_value = p[i].pbest.fitness_value;
		}
	#endif

	#ifdef MIN
		if(p[i].pbest.fitness_value < gbest.fitness_value) {
			for(int j = 0; j < DIM; j++) {
				gbest.dimension[j] = p[i].pbest.dimension[j]; 
			}
			gbest.fitness_value = p[i].pbest.fitness_value;
		}
	#endif
	}

	return;
}

void move(vector<particle>& p,global_best gbest) {
	for(int i = 0; i < SIZE; i++) {
		for(int j = 0; j < DIM; j++) {
			p[i].speed[j] = W * p[i].speed[j] + C1 * (double)rand()/(RAND_MAX + 1.0) * (p[i].pbest.dimension[j] - p[i].dimension[j])  + C2 * (double)rand()/(RAND_MAX + 1.0) * (gbest.dimension[j] - p[i].dimension[j]);
			if(p[i].speed[j] > S_UPPER)
				p[i].speed[j] = S_UPPER;
			if(p[i].speed[j] < S_LOWER)
				p[i].speed[j] = S_LOWER;

			p[i].dimension[j] += p[i].speed[j]; 

			if(p[i].dimension[j] > UPPER) {
				double temp = (double)rand()/(RAND_MAX + 1.0);
				p[i].dimension[j] = temp * (UPPER-LOWER)+LOWER;
			}
			if(p[i].dimension[j] < LOWER) {
				double temp = (double)rand()/(RAND_MAX + 1.0);
				p[i].dimension[j] = temp * (UPPER-LOWER)+LOWER;
			}
		}
	}

	return;
}

void fitness_cal(vector<particle>& p) {
	for(int i = 0; i < SIZE; i++) {
		p[i].fitness_value = fitness(p[i]);

	#ifdef MAX
		if(p[i].fitness_value > p[i].pbest.fitness_value) {
			for(int j = 0; j <DIM; j++) {
				p[i].pbest.dimension[j] = p[i].dimension[j];
			}
			p[i].piest.fitness_value = p[i].fitness_value;
		}
	#endif

	#ifdef MIN
		if(p[i].fitness_value < p[i].pbest.fitness_value) {
			for(int j = 0; j < DIM; j++) {
				p[i].pbest.dimension[j] = p[i].dimension[j];
			}
			p[i].pbest.fitness_value = p[i].fitness_value;
		}
	#endif
	}

	return;
}

int main() {
	vector<particle> p(SIZE);
	global_best gbest;

	struct timespec t_start, t_end;
	double elapsedTime, totalTime = 0;

	initialize(p);

	clock_gettime(CLOCK_REALTIME, &t_start);
	find_gbest(p , gbest);

	for(int i = 0; i < ITERATION; i++) {
	#ifdef DEBUG
		for(int i = 0; i < SIZE; i++) {
			for(int j = 0; j < DIM; j++) {
				cout << "particle " << i << " dim " << j << " : " << p[i].dimension[j] << endl;
			}
			cout << "particle " << i << " fitness value : " << p[i].fitness_value << endl;
		}
		cout<<endl;
	#endif

		move(p , gbest);
		fitness_cal(p);
		find_gbest(p , gbest);

	#ifdef DEBUG
		cout << "ITERATION " << i << " : " << endl;
		cout << "local_best : " << gbest.fitness_value << endl;
		for(int j = 0; j < DIM; j++) {
			cout << "dimension " << j << " : " << gbest.dimension[j] << endl;
		}
	#endif
	}

	clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("CPU time: %13f msec\n", elapsedTime);

	printf("global best : %13lf\n", gbest.fitness_value);
	// cout << "global best : " << gbest.fitness_value << endl;
	// for(int j = 0; j < DIM; j++) {
	// 	cout << "dimension " << j << " : " << gbest.dimension[j] << endl;
	// }

	FILE *f = fopen(FILE_PATH, "a");
	fprintf(f, "%.15lf,%lf\n", gbest.fitness_value, elapsedTime);
	fclose(f);
	return 0;
}
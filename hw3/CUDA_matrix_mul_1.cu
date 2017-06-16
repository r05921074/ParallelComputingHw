#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

//#define DEBUG

#define L1 1024
#define L2 1024
#define L3 1024

/* ========== Multiple block, Multiple threads ========== */
/* ========== Can change different matrix length and width ========== */
/* ========== B matrix doen't transposed ========== */
/* ========== fixed block dimension as 32 * 32 ========== */
/* ========== Max array length: 1024 due to MaxThread per side is 1024 ========== */

__global__ void MatMulKernel(float *Ad, float *Bd, float *Cd);
void MatMul(float *A, float *B, float *C);


int main(int argc, char *argv[])
{
    float *A, *B, *C, *AxB;
    int pass = 1;
    A = (float *)calloc(L1 * L2, sizeof(float));
    B = (float *)calloc(L2 * L3, sizeof(float));
    C = (float *)calloc(L1 * L3, sizeof(float));
    AxB = (float *)calloc(L1 * L3, sizeof(float));
  
    /* ========== Assign values to array A and B ========== */  
    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L2; ++j) {
            A[i * L2 + j] = rand() % 30;
        }
    }
    for (int i = 0; i < L2; ++i) {
        for (int j = 0; j < L3; ++j) {
            B[i * L3 + j] = rand() % 30;
        }
    }

#ifdef DEBUG
    printf("Matrix A:\n");
    for (int i = 0; i < L1; i++) {
        for (int j = 0; j < L2; j++) {
            printf("%3.0f", A[i * L2 + j]);
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < L2; i++) {
        for (int j = 0; j < L3; j++) {
            printf("%3.0f", B[i * L3 + j]);
        }
        printf("\n");
    }
#endif
    
    /* ========== Calculate correct answers by CPU ========== */
    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);
    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L3; ++j) {
            for (int k = 0; k < L2; ++k) {
                AxB[i * L3 + j] += A[i * L2 + k] * B[k * L3 + j];
            }
        }
    }
    gettimeofday(&endtime, NULL);
    double executime;
    executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
    executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
    printf("CPU time: %13lf msec\n", executime);

#ifdef DEBUG
    printf("Matrix AxB:\n");
    for (int i = 0; i < L1; i++) {
        for (int j = 0; j < L3; j++) {
            printf("%5.0f", AxB[i * L3 + j]);
        }
        printf("\n");
    }
#endif
    
    /* ========== Calculate answers by GPU ========== */
    MatMul((float *)A, (float *)B, (float *)C);

#ifdef DEBUG
    printf("Matrix C:\n");
    for (int i = 0; i < L1; i++) {
        for (int j = 0; j < L3; j++) {
            printf("%5.0f", C[i * L3 + j]);
        }
        printf("\n");
    }
#endif
    
    /* ========== Check if answers correct ========== */
    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L3; ++j) {
            if(AxB[i * L3 + j] != C[i * L3 + j]) {
				printf("AxB[%d][%d] = %2.0f   C[%d][%d] = %2.0f\n", i, j, AxB[i * L3 + j], i, j, C[i * L3 + j]);
                pass = 0;
            }
        }
    }
    
    printf("Test %s\n", (pass)?"PASSED":"FAILED");

    free(A);
    free(B);
    free(C);
    free(AxB);
    
    return 0;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float *Ad, float *Bd, float *Cd)
{
    // Thread row and column within matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < L1 && col < L3) {
        float Cvalue = 0;

        for (int k = 0; k < L2; k++) {
            float Aval = Ad[row * L2 + k];
            float Bval = Bd[k * L3 + col];
            Cvalue += Aval * Bval;
        }

        Cd[row * L3 + col] = Cvalue;
    }
}

/* ========== Matrix multiplication - Host code ========== */
void MatMul(float *A, float *B, float *C)
{
    size_t size_1 = L1 * L2 * sizeof(float);
    size_t size_2 = L2 * L3 * sizeof(float);
    size_t size_3 = L1 * L3 * sizeof(float);
    float *Ad, *Bd, *Cd;
    
    /* ========== Allocate and Load A, B to device memory ========== */
    cudaMalloc((void **)&Ad, size_1);
    cudaMemcpy(Ad, A, size_1, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&Bd, size_2);
    cudaMemcpy(Bd, B, size_2, cudaMemcpyHostToDevice);
    
    /* ========== Allocate C on the device ========== */
    cudaMalloc((void **)&Cd, size_3);
    
    /* ========== Setup the execution configuration ========== */
    int GridDim_x = (L3 + 31) / 32, GridDim_y = (L1 + 31) / 32;
    dim3 dimGrid(GridDim_x, GridDim_y);
    dim3 dimBlock(32, 32);

    /* ========== Get start time event ========== */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    /* ========== Invoke kernel ========== */
    MatMulKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Cd);
    cudaError_t cuda_err = cudaGetLastError();
    if ( cudaSuccess != cuda_err ){
        printf("before kernel call: error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    
    /* ========== Get stop time event ========== */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    /* ========== Compute execution time ========== */
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    /* ========== Read C from device memory ========== */
    cudaMemcpy(C, Cd, size_3, cudaMemcpyDeviceToHost);
    
    /* ========== Free device memory ========== */
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}

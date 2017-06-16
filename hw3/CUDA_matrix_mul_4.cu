#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

//#define B_T
//#define DEBUG

#define L1 383
#define L2 103
#define L3 993
#define TILE_WIDTH  32

/* ========== Multiple block, Multiple threads ========== */
/* ========== Tile multiplication, Shared memory ========== */
/* ========== Can change different matrix length and width (with bug now, only accept multiply of TILE_WIDTH)========== */
/* ========== B matrix doesn't transposed ========== */
/* ========== fixed block dimension as 32 * 32 ========== */
/* ========== Max array length: 1024 due to MaxThread per side is 1024 ========== */

__device__ float GetElement(float *matrix, int row, int col, int width);
__device__ void SetElement(float *matrix, int row, int col, int width, float value);
__device__ float *GetSubMatrix(float *matrix, int blockrow, int blockcol, int width);
__global__ void MatMulKernel(float *Ad, float *Bd, float *Cd);
void MatMul(float *A, float *B, float *C);


int main(int argc, char *argv[])
{
    int pass = 1;

    float *A = (float *)calloc(L1 * L2, sizeof(float));
    float *B = (float *)calloc(L2 * L3, sizeof(float));
    float *C = (float *)calloc(L1 * L3, sizeof(float));
    float *AxB = (float *)calloc(L1 * L3, sizeof(float));
    
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

#ifdef B_T
    for (int i = 0; i < L2; i++) {
        for (int j = 0; j < L3; j++) {
            //B_t[j * L2 + i] = B[i * L3 + j];
        }
    }
#endif

#ifdef DEBUG
    // printf("Matrix B_t:\n");
    // for (int i = 0; i < L3; i++) {
    //     for (int j = 0; j < L2; j++) {
    //         printf("%5.0f", B_t[i * L2 + j]);
    //     }
    //     printf("\n");
    // }
#endif

    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L3; ++j) {
            for (int k = 0; k < L2; ++k) {
                #ifdef B_T
                //AxB[i * L3 + j] += A[i * L2 + k] * B_t[j * L2 + k];
                #endif
                #ifndef B_T
                AxB[i * L3 + j] += A[i * L2 + k] * B[k * L3 + j];
                #endif
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
                //printf("AxB[%d][%d] = %2.0f   C[%d][%d] = %2.0f\n", i, j, AxB[i * L3 + j], i, j, C[i * L3 + j]);
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

// Get a matrix element
__device__ float GetElement(float *matrix, int row, int col, int width)
{
    return *(matrix + row*width + col);
}

// Set a matrix element
__device__ void SetElement(float *matrix, int row, int col, int width, float value)
{
    *(matrix + row*width + col) = value;
}

// Get the TILE_WIDTHxTILE_WIDTH sub-matrix matsub of matrix that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of matrix
__device__ float *GetSubMatrix(float *matrix, int blockrow, int blockcol, int width)
{
    return (matrix + blockrow*TILE_WIDTH*width + blockcol*TILE_WIDTH);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float *Ad, float *Bd, float *Cd)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int globalRow = blockRow * blockDim.y + row;
    int globalCol = blockCol * blockDim.x + col;

    int iter = (L2 + TILE_WIDTH - 1) / TILE_WIDTH;
    int residue = L2 % TILE_WIDTH;

    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];
    
    float Cvalue = 0;
    for (int m = 0; m < iter; ++m) {
        if (globalRow < L1 && (m * TILE_WIDTH + col) < L2) 
            shared_A[row][col] = Ad[globalRow * L2 + (m * TILE_WIDTH + col)];
        else
            shared_A[row][col] = 0;

        if ((m * TILE_WIDTH + row) < L2 && globalCol < L3) 
            shared_B[row][col] = Bd[(m * TILE_WIDTH + row) * L3 + globalCol];
        else
            shared_B[row][col] = 0;

        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            float Aelement = shared_A[row][k];
            float Belement = shared_B[k][col];
            Cvalue += Aelement * Belement;
        }

        __syncthreads();
    }

    if (globalRow < L1 && globalCol < L3)
        Cd[globalRow * L3 + globalCol] = Cvalue;
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
    int GridDim_x = (L3 + TILE_WIDTH - 1) / TILE_WIDTH;
    int GridDim_y = (L1 + TILE_WIDTH - 1) / TILE_WIDTH;
    printf("%d, %d\n", GridDim_x, GridDim_y);
    dim3 dimGrid(GridDim_x, GridDim_y);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    /* ========== Get start time event ========== */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    /* ========== Invoke kernel ========== */
    cudaError_t cuda_err = cudaGetLastError();
    if ( cudaSuccess != cuda_err ){
        printf("before kernel call: error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    MatMulKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Cd);
    cuda_err = cudaGetLastError();
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
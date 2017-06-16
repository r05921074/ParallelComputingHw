#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define VALUE_MAX 10000000
#define BLOCK_LENGTH 32

void FindBestMatch(IplImage *sourceImg, IplImage *patternImg, int *result);
__global__ void MatchKernel(int *d_src, int *d_ptn, int *d_res, int src_h, int src_w, int ptn_h, int ptn_w, int res_h, int res_w, int src_ws, int ptn_ws);

struct match {
	int bestRow;
	int bestCol;
	int bestSAD;
}position;

struct size {
	int height;
	int width;
	int nchannels;
}datasize;

int main( int argc, char** argv ) {
	IplImage* sourceImg; 
	IplImage* patternImg; 

	int minSAD = VALUE_MAX;
	int x, y;
	CvPoint pt1, pt2;

	struct timespec t_start, t_end;
	double elapsedTime;

	int result_height;
	int result_width;
	int *result;

	if( argc != 3 ) {
		printf("Using command: %s source_image search_image\n",argv[0]);
		exit(1);
	}

	if((sourceImg = cvLoadImage(argv[1], 0)) == NULL) {
		printf("%s cannot be openned\n",argv[1]);
		exit(1);
	}

	printf("height of sourceImg:%d\n",sourceImg->height);
	printf("width of sourceImg:%d\n",sourceImg->width);
	printf("size of sourceImg:%d\n",sourceImg->imageSize);

	if((patternImg = cvLoadImage( argv[2], 0)) == NULL) {
		printf("%s cannot be openned\n",argv[2]);
		exit(1);
	}

	printf("height of sourceImg:%d\n",patternImg->height);
	printf("width of sourceImg:%d\n",patternImg->width);
	printf("size of sourceImg:%d\n",patternImg->imageSize);    

	//allocate memory on CPU to store SAD results
	result_height = sourceImg->height - patternImg->height + 1;
	result_width = sourceImg->width - patternImg->width + 1;
	result = (int *)malloc(result_height * result_width * sizeof(int));

	// start time
	clock_gettime( CLOCK_REALTIME, &t_start);

	FindBestMatch(sourceImg, patternImg, result);

	// stop time
	clock_gettime( CLOCK_REALTIME, &t_end);

	// compute and print the elapsed time in millisec
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%lf ms\n", elapsedTime);

	for (y=0; y < result_height; y++) {
		for (x=0; x < result_width; x++) {
			if (minSAD > result[y * result_width + x]) {
				minSAD = result[y * result_width + x];

				// give me VALUE_MAX
				position.bestRow = y;
				position.bestCol = x;
				position.bestSAD = result[y * result_width + x];

			}
		}
	}
	printf("minSAD is %d\n", minSAD);

	//setup the two points for the best match
	pt1.x = position.bestCol;
	pt1.y = position.bestRow;
	pt2.x = pt1.x + patternImg->width;
	pt2.y = pt1.y + patternImg->height;
	printf("pt1.x: %d, pt1.y: %d, pt2.x: %d, pt2.y: %d\n", pt1.x, pt1.y, pt2.x, pt2.y);

	// Draw the rectangle in the source image
	cvRectangle(sourceImg, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0);
	cvNamedWindow("sourceImage", 1);
	cvShowImage("sourceImage", sourceImg);
	cvNamedWindow("patternImage", 1);
	cvShowImage("patternImage", patternImg);
	cvWaitKey(0); 
	cvDestroyWindow("sourceImage");
	cvReleaseImage(&sourceImg);
	cvDestroyWindow("patternImage");
	cvReleaseImage(&patternImg);

	return 0;
}

void FindBestMatch(IplImage *sourceImg, IplImage *patternImg, int *result) {
	int *src, *ptn;
	src = (int *)malloc(sourceImg->height * sourceImg->widthStep * sizeof(int));
	ptn = (int *)malloc(patternImg->height * patternImg->widthStep * sizeof(int));
	for (int i = 0; i < sourceImg->height; i++) {
		for (int j = 0; j < sourceImg->widthStep; j++) {
			src[i * sourceImg->widthStep + j] = sourceImg->imageData[i * sourceImg->widthStep + j];
		}
	}
	for (int i = 0; i < patternImg->height; i++) {
		for (int j = 0; j < patternImg->widthStep; j++) {
			ptn[i * patternImg->widthStep + j] = patternImg->imageData[i * patternImg->widthStep + j];
		}
	}

	int *i_res;
	int res_height = sourceImg->height - patternImg->height +1;
	int res_width = sourceImg->width - patternImg->width + 1;
	i_res = (int *)malloc(res_height * res_width * sizeof(int));

	size_t size_src = sourceImg->height * sourceImg->widthStep * sizeof(int);
	size_t size_ptn = patternImg->height * patternImg->widthStep * sizeof(int);
	size_t size_res = res_height * res_width * sizeof(int);
	int *d_src, *d_ptn, *d_res;

	cudaMalloc((void **)&d_src, size_src);
	cudaMemcpy(d_src, src, size_src, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_ptn, size_ptn);
	cudaMemcpy(d_ptn, ptn, size_ptn, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_res, size_res);

	int dimGrid_x = (res_width + BLOCK_LENGTH - 1) / BLOCK_LENGTH;
	int dimGrid_y = (res_height + BLOCK_LENGTH - 1) / BLOCK_LENGTH;
	dim3 dimGrid(dimGrid_x, dimGrid_y);
	dim3 dimBlock(BLOCK_LENGTH, BLOCK_LENGTH);
	MatchKernel<<<dimGrid, dimBlock>>>(d_src, d_ptn, d_res, sourceImg->height, sourceImg->width, patternImg->height, patternImg->width, res_height, res_width, sourceImg->widthStep, patternImg->widthStep);

	cudaMemcpy(i_res, d_res, size_res, cudaMemcpyDeviceToHost);
	for (int i = 0; i < res_height; i++) {
		for (int j = 0; j < res_width; j++) {
			result[i * res_width + j] = i_res[i * res_width + j];
		}
	}
}

__global__ void MatchKernel(int *d_src, int *d_ptn, int *d_res, int ptn_h, int ptn_w, int res_h, int res_w, int src_ws, int ptn_ws) {
	int global_Row = blockIdx.y * blockDim.y + threadIdx.y;
	int global_Col = blockIdx.x * blockDim.x + threadIdx.x;
	float SAD = 0.0;
	float src_pixel = 0.0, ptn_pixel = 0.0;
	float diff = 0.0;

	if (global_Row < res_h && global_Col < res_w) {
		for (int i = 0; i < ptn_h; i++) {
			for (int j = 0; j < ptn_w; j++) {
				src_pixel = d_src[(global_Row+i)*src_ws + (global_Col+j)];
				ptn_pixel = d_ptn[i*ptn_ws + j];
				diff = src_pixel - ptn_pixel;
				if (diff < 0) {
					SAD += (-1) * diff;
				}
				else {
					SAD += diff;
				}
			}
		}
		d_res[global_Row*res_w + global_Col] = SAD;
	}
}
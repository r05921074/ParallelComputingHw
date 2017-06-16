#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define VALUE_MAX 10000000

struct match {
	int bestRow;
	int bestCol;
	int bestSAD;
}position;

int main( int argc, char** argv ) {
	IplImage* sourceImg; 
	IplImage* patternImg; 

	int minSAD = VALUE_MAX;
	int SAD;
	int x, y, i, j;

	uchar* ptr;
	uchar p_sourceIMG, p_patternIMG;
	CvPoint pt1, pt2;

	struct timespec t_start, t_end;
	double elapsedTime;

	int result_height;
	int result_width;
	int *host_result;

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
	host_result=(int*)malloc(result_height * result_width * sizeof(int));

	// printf("SourceImg\n");
	// for (int i = 0; i < 20; i++) {
	// 	for (int j = 0; j < 20; j++) {
	// 		printf("%d, ", sourceImg->imageData[i * sourceImg->widthStep + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n");

	// printf("PatternImg\n");
	// for (int i = 0; i < 20; i++) {
	// 	for (int j = 0; j < 20; j++) {
	// 		printf("%d, ", patternImg->imageData[i * patternImg->widthStep + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n");

	// start time
	clock_gettime( CLOCK_REALTIME, &t_start);

	for (y = 0; y < result_height; y++) {
		for(x = 0; x < result_width; x++) {
			SAD = 0.0;
			// loop through the template image
			for (j = 0; j < patternImg->height; j++){
				for (i = 0; i < patternImg->width; i++) {
					p_sourceIMG = sourceImg->imageData[(y+j) * sourceImg->widthStep+x+i];
					p_patternIMG = patternImg->imageData[j * patternImg->widthStep+i];
					SAD += abs(p_sourceIMG - p_patternIMG);
				}
			}
			host_result[y * result_width + x]=SAD;
		}
	}

	for (int i = 0; i < 15; i++) {
		for (int j = 0; j < 15; j++) {
			printf("%d, ", host_result[i * result_width + j]);
		}
		printf("\n");
	}
	printf("\n");


	// stop time
	clock_gettime( CLOCK_REALTIME, &t_end);

	// compute and print the elapsed time in millisec
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%lf ms\n", elapsedTime);

	for (y=0; y < result_height; y++) {
		for (x=0; x < result_width; x++) {
			if (minSAD > host_result[y * result_width + x]) {
				minSAD = host_result[y * result_width + x];

				// give me VALUE_MAX
				position.bestRow = y;
				position.bestCol = x;
				position.bestSAD = host_result[y * result_width + x];

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
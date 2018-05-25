
#include <stdlib.h>
#include <stdio.h>
//
// these libraries are for CUDA RNG
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// define number of cores to use
#define N 20

// define number of trials to use
#define M 10000000

// we will be using unsigned ints because we want a random positive number between 0 and 1
// kernel for generating random number then executing monte carlo method
__global__ void mcarlo( float *d_area, int totarea ) {
	
	int i;
	totarea = 0;
	unsigned int seed = threadIdx.x;
	curandState s;
	
	// seed a random number generator
	curand_init(seed, 0, 0, &s);

	// for loop to implement monte carlo
	for(i = 0; i < M; ++i) {
		int ind = blockIdx.x * blockDim.x + threadIdx.x;
		float x = curand_uniform(&s);
		float y = curand_uniform(&s);
		if( x*x + y*y <= 1.0f) {
			d_area[ind] += 1;
		}		
	}

	// sum up elements in array
	i = 0;
	while( i < M )
        {
                totarea += d_area[i];
                i++;
        }

}

// main function
int main( void ) {
	// initiate host variables
	int h_tot;
	int totarea = 0;

	//allocate host memory (unnecessary?)
	//h_area = (float*)  malloc(M * sizeof(float));

	// initiate device
	int deviceid = 0;
	int devCount;
	cudaGetDeviceCount(&devCount);
	if(deviceid<devCount) cudaSetDevice(deviceid);
	else return(1);

	// define and allocate memory on the device
	float *d_area;
	cudaMalloc(&d_area, M*sizeof(float));	

	// execute kernel to implement mcarlo
	mcarlo<<<N , 1>>>( d_area, totarea );

	// transfer result back to host
	cudaMemcpy(&h_tot, &totarea, sizeof(int), cudaMemcpyDeviceToHost);
	
	// calculate pi
	int PI = 4 * h_tot / N;

	// display result
	printf("\nPI is %f\n", PI);

	// free memory 
	cudaFree(d_area);
	
	return 0;
}

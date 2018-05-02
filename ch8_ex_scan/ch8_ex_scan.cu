#include "gpuerrchk.cuh"
#include "real.h"
#define SECTION_SIZE 1024 

//input size=SECTION_SIZE is twice block size, so that we can have 2 threads per element to be summed
__global__ void ch8_ex_scan_kernel(real* X, real* Y, int inputsize){
	__shared__ real XY[SECTION_SIZE];
	int i=2*blockIdx.x*blockDim.x+threadIdx.x;
	if (i < inputsize) XY[threadIdx.x]=X[i];
	if (i + blockDim.x <inputsize) XY[threadIdx.x+blockDim.x]=X[i+blockDim.x];	
	
	//up-sweep
	for (int stride=1; stride <= blockDim.x; stride*=2){
		__syncthreads();
		int index= 2*(threadIdx.x+1)*stride -1;
		if (index< SECTION_SIZE) XY[index]+=XY[index-stride];
	}


	//down-sweep
	for (int stride=SECTION_SIZE/2; stride>=1; stride/=2){
		__syncthreads();
		int index = 2*(threadIdx.x+1)*stride-1;
		if (index+stride < SECTION_SIZE){
			XY[index+stride]+=XY[index];
		}
	}
	__syncthreads();
	if (i< inputsize) Y[i]= XY[threadIdx.x];
	if (i+blockDim.x < inputsize) Y[i+blockDim.x]=XY[threadIdx.x+blockDim.x]; 
}

void ch8_ex_scan(real* d_X, real* d_Y,int inputsize){
	ch8_ex_scan_kernel<<<1,SECTION_SIZE/2>>>(d_X,d_Y,inputsize);
	gpuErrchk(cudaPeekAtLastError());
}

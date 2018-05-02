#include "real.h"
#include "math.h"

#define SECTION_SIZE 512
//ACTUALLY THIS SEEMS WRONG: WE DO NOT KNOW THE ORDER OF OPERATIONS OF THE ADDING.  NEED TO DOUBLE BUUFER THE ARRAY XY TO GUARATEE THAT THIS WORKS.
__global__ void ksscan_kernel(real* X, real* Y, int inputsize){
	__shared__ real XY[SECTION_SIZE];
	int i =blockIdx.x*blockDim.x+threadIdx.x;
	if (i < inputsize){
		XY[threadIdx.x]=X[i]; 
		for (int stride=1; stride<blockDim.x; stride*=2 ){
			__syncthreads();
			if (threadIdx.x >= stride) XY[threadIdx.x]+= XY[threadIdx.x-stride];
		}
		Y[i]=XY[threadIdx.x];
	}
}

void ksscan(real* d_X, real* d_Y,int inputsize){
	ksscan_kernel<<<ceil(inputsize/ (real) SECTION_SIZE),SECTION_SIZE>>>(d_X,d_Y,inputsize);
}




#include <math.h>
#include "gpuerrchk.cuh"
#include "real.h"
#include "next_pow_2.h"
#include <assert.h>
#include <iostream>
__global__ void ch8_longer_scan_kernel1(real* X, real* Y,real* S,unsigned int S_length, unsigned int treesize){
	extern __shared__ real XY[];
	int i=2*blockIdx.x*blockDim.x+threadIdx.x;
	 XY[threadIdx.x]=X[i];
	 XY[threadIdx.x+blockDim.x]=X[i+blockDim.x];	
	
	//up-sweep
	for (int stride=1; stride <= blockDim.x; stride*=2){
		__syncthreads();
		int index= 2*(threadIdx.x+1)*stride -1;
		if (index< treesize) XY[index]+=XY[index-stride];
	}


	//down-sweep
	for (int stride=treesize/2; stride>=1; stride/=2){
		__syncthreads();
		int index = 2*(threadIdx.x+1)*stride-1;
		if (index+stride < treesize){
			XY[index+stride]+=XY[index];
		}
	}
	__syncthreads();
	 Y[i]= XY[threadIdx.x];
	 Y[i+blockDim.x]= XY[threadIdx.x+blockDim.x]; 
	 if(threadIdx.x == blockDim.x-1)
		 S[blockIdx.x]= XY[treesize-1];
}

//performs an in-place scan on S
//full s length must be a power of 2
__global__ void ch8_longer_scan_kernel2(real* S, unsigned int full_S_length){
	extern __shared__ real XY[];
	int i=2*blockIdx.x*blockDim.x+threadIdx.x;
	XY[threadIdx.x]=S[i];
	XY[threadIdx.x+blockDim.x]=S[i+blockDim.x];	
	
	//up-sweep
	for (int stride=1; stride <= blockDim.x; stride*=2){
		__syncthreads();
		int index= 2*(threadIdx.x+1)*stride -1;
		if (index< full_S_length) XY[index]+=XY[index-stride];
	}


	//down-sweep
	for (int stride=full_S_length/2; stride>=1; stride/=2){
		__syncthreads();
		int index = 2*(threadIdx.x+1)*stride-1;
		if (index+stride < full_S_length){
			XY[index+stride]+=XY[index];
		}
	}
	__syncthreads();
	 S[i]= XY[threadIdx.x];
	 S[i+blockDim.x]=XY[threadIdx.x+blockDim.x]; 

}


__global__ void ch8_longer_scan_kernel3(real* Y, real* S){	
	int i=2*blockIdx.x*blockDim.x+threadIdx.x;
	if (blockIdx.x>0) {
		Y[i]+=S[blockIdx.x-1];
		Y[i+blockDim.x]+=S[blockIdx.x-1];
	}
}

//NEED TO DO MULTISTAGE SCAN ENTIRELY ON DEVICE, taking device input and writing device output

//treesize is assumed to be a power of 2.  d_X and d_Y are assumed to be of length length. 
//also assume treesize*(S_length)=length
//also assume d_S is small enough to be scanned by one thread block.
//also assume d_S points to an array with length equal to S_length rounded to the next power of 2
void ch8_longer_scan(real* d_X, real* d_Y,real* d_S, size_t length, unsigned int S_length, unsigned int treesize){
	cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop,0); //assume we are using device 0
	size_t share_mem=dev_prop.sharedMemPerBlock;
	int thread_limit= dev_prop.maxThreadsPerBlock;
	size_t max_per_block=share_mem/sizeof(real);
	assert(treesize<=max_per_block && treesize<=2*thread_limit);
	assert(treesize*(S_length)==length);
	ch8_longer_scan_kernel1<<<S_length, treesize/2, treesize*sizeof(real)>>>(d_X, d_Y, d_S, S_length, treesize);
	gpuErrchk(cudaPeekAtLastError());
	
	//debugging
//	real Y[2048];
//	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*2048,cudaMemcpyDeviceToHost));
//	for(int i=0; i<2048; i++)
//		std::cout << "i=" << i << " Y[i]=" << Y[i] <<std::endl;




	unsigned int full_S_length=next_pow_2(S_length);
	assert(full_S_length <= max_per_block && full_S_length<=2*thread_limit);
	ch8_longer_scan_kernel2<<<1,full_S_length/2,sizeof(real)*full_S_length>>>(d_S,full_S_length);	
	gpuErrchk(cudaPeekAtLastError());	
	ch8_longer_scan_kernel3<<<S_length,treesize/2, treesize*sizeof(real)>>>(d_Y,d_S);	
	gpuErrchk(cudaPeekAtLastError());	
}

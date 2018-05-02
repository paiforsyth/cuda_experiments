#include <math.h>
#include "gpuerrchk.cuh"
#include "real.h"
#include "next_pow_2.h"
#include <assert.h>
#include <iostream>
__global__ void ch8_stream_scan_kernel1(real* X, real* Y,real* S,int* flags,int* DCounter, unsigned int S_length, unsigned int treesize){
	extern __shared__ real XY[];
	
	__shared__ int sbid;
	if (threadIdx.x == 0)
		sbid=atomicAdd(DCounter, 1);
	__syncthreads();
	const int bid=sbid;

	int i=2*bid*blockDim.x+threadIdx.x;
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

	//domino
	__syncthreads(); 
	__shared__ float previous_sum;
	 if (threadIdx.x == blockDim.x-1  ){
		if (bid > 0){
			while(atomicAdd(&flags[bid-1],0 ) == 0 ) {;}
			previous_sum=S[bid-1];
			S[bid]=previous_sum+XY[2*blockDim.x-1];
			__threadfence();
			atomicAdd(&flags[bid],1);
		}
		else{
			previous_sum=0;
			S[0]=XY[2*blockDim.x-1];
			__threadfence();
			atomicAdd(&flags[0],1);
		}
	 }

	 __syncthreads();
	 XY[threadIdx.x]+=previous_sum;
	 XY[threadIdx.x+blockDim.x]+=previous_sum;
	 Y[i]= XY[threadIdx.x];
	 Y[i+blockDim.x]= XY[threadIdx.x+blockDim.x]; 
}




//treesize is assumed to be a power of 2.  d_X and d_Y are assumed to be of length length. 
//also assume treesize*(S_length)=length
//also assume d_S is small enough to be scanned by one thread block.
//also assume d_S points to an array with length equal to S_length rounded to the next power of 2
void ch8_stream_scan(real* d_X, real* d_Y,real* d_S,int* flags, int* DCounter, size_t length, unsigned int S_length, unsigned int treesize){
	cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop,0); //assume we are using device 0
	size_t share_mem=dev_prop.sharedMemPerBlock;
	int thread_limit= dev_prop.maxThreadsPerBlock;
	size_t max_per_block=share_mem/sizeof(real);
	assert(treesize<=max_per_block && treesize<=2*thread_limit);
	assert(treesize*(S_length)==length);
	ch8_stream_scan_kernel1<<<S_length, treesize/2, treesize*sizeof(real)>>>(d_X, d_Y, d_S, flags, DCounter, S_length, treesize);
	gpuErrchk(cudaPeekAtLastError());
}

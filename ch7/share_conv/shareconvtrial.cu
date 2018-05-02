#include "assert.h"
#include "real.h"
#include <iostream>
#include "gpuerrchk.cuh"
#include "math.h"

#define MAX_MASK_WIDTH 10
#define TILE_SIZE 1000
__device__ __constant__ float d_M[1000];

__global__ void share_conv_kernel(real* A, real* P, int mask_width, int width){
	__shared__ real A_s[TILE_SIZE];	
	A_s[threadIdx.x]=A[blockIdx.x*blockDim.x+threadIdx.x];
	__syncthreads();	
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int this_tile_start_point = blockIdx.x*blockDim.x;
	int next_tile_start_point = (blockIdx.x+1)*blockDim.x;
	int mask_start_point= i-mask_width/2;	
	real Pvalue=0; //mask width is assumed odd  So there are mask_width integers in [-mask_width/2, mask_width/2]
	for (int j=0; j< mask_width; j++){
		int mask_index= mask_start_point +j;
		if( mask_index< 0  || mask_index >= width)
			continue;
		if(	mask_index >=this_tile_start_point && mask_index < next_tile_start_point)
			Pvalue+=A_s[threadIdx.x+j -mask_width/2]*d_M[j];
		else
			Pvalue+=A[mask_index]*d_M[j];
	}
	P[i]=Pvalue;
}

void share_conv(real* A,float* M, real* P, int mask_width, int width ){
	real* d_A;
	real* d_P;
	gpuErrchk(cudaMalloc((void**)&d_A, sizeof(real)*width ));
	gpuErrchk(cudaMemcpy(d_A, A, sizeof(real)*width, cudaMemcpyHostToDevice )  );
	gpuErrchk(cudaMemcpyToSymbol(d_M, M, sizeof(real)*mask_width )  );
	gpuErrchk(cudaMalloc((void**)&d_P, sizeof(real)*width ));
	int blocksize=512;
	share_conv_kernel<<<ceil(width/ (real)blocksize),blocksize >>>(d_A,  d_P, mask_width, width);
	gpuErrchk(cudaMemcpy(P, d_P, sizeof(real)*width, cudaMemcpyDeviceToHost )  );
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaFree(d_A ) );
	gpuErrchk(cudaFree(d_P ) );
}

/*void trial(){
	constexpr int asize=10^5;
	constexpr int bsize=1000;
	real A[asize];
	for(int i=0; i< asize; i++){
		A[i]=1;
	}
	real M[bsize];
	for (int i=0; i<bsize; ++i){
		M[i]=i;	
	}
	real P[asize];
	share_conv(A,M,P,bsize,asize);
}
*/


	


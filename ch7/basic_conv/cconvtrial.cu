#include "assert.h"
#include "real.h"
#include <iostream>
#include "gpuerrchk.cuh"
#include "math.h"

#define MAX_MASK_WIDTH 10
__device__ __constant__ float d_M[1000];

__global__ void constant_conv_kernel(real* A, real* P, int mask_width, int width){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	real Pvalue=0; //mask width is assumed odd  So there are mask_width values in [-mask_width/2, mask_width/2]
	for(int j=i-mask_width/2; j<=i+mask_width/2; ++j){
		if (j>=0 && j<width)
			Pvalue+= A[j]*d_M[j-(i-mask_width/2)];

	}
	P[i]=Pvalue;
}

void constant_conv(real* A,float* M, real* P, int mask_width, int width ){
	real* d_A;
	real* d_P;
	gpuErrchk(cudaMalloc((void**)&d_A, sizeof(real)*width ));
	gpuErrchk(cudaMemcpy(d_A, A, sizeof(real)*width, cudaMemcpyHostToDevice )  );
	gpuErrchk(cudaMemcpyToSymbol(d_M, M, sizeof(real)*mask_width )  );
	gpuErrchk(cudaMalloc((void**)&d_P, sizeof(real)*width ));
	int blocksize=512;
	constant_conv_kernel<<<ceil(width/ (real)blocksize),blocksize >>>(d_A,  d_P, mask_width, width);
	gpuErrchk(cudaMemcpy(P, d_P, sizeof(real)*width, cudaMemcpyDeviceToHost )  );
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaFree(d_A ) );
	gpuErrchk(cudaFree(d_P ) );

}
void trial(){
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
	constant_conv(A,M,P,bsize,asize);
}

int main(){
	trial();
	
}

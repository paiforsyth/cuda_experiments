#include <math.h>
#include <iostream>
//compute  vector sum C=A+B
//each thread performs one pair-wise addition.
__global__
void vecAddKernel(float* A, float* B, float* C, int n){
	int i=blockDim.x*blockIdx.x + threadIdx.x;
	if (i<n) C[i]= A[i]+ B[i];
}
void vecAdd(float* A, float* B, float* C,int n){
	int size=n*sizeof(float);
	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc( (void**)&d_A, size);
	cudaMalloc( (void**)&d_B, size);
	cudaMalloc( (void**)&d_C, size);
	cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,A,size,cudaMemcpyHostToDevice);
	vecAddKernel<<<ceil(n/256.0), 256>>>(d_A,d_B,d_C,n);
	
	cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
	cudaFree(d_A);

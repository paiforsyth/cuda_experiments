#include "ksscan.cuh"
#include "real.h"
#include "gpuerrchk.cuh"
#include "assert.h"
#include <iostream>
void test(){
	real X[]={3, 1, 7, 0 ,4 ,1 ,6, 3};
	real Y[8];
	real* d_X;
	real* d_Y;

	gpuErrchk(cudaMalloc((void**) &d_X,sizeof(real)*8));
	gpuErrchk(cudaMemcpy(d_X,X,sizeof(real)*8,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_Y,sizeof(real)*8 ));
	ksscan(d_X,d_Y,8);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*8,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));
	assert(Y[3]==11);
	assert(Y[7]==25);
}

int main(){
	test();
	std::cout << "success!!!"<<std::endl;
}

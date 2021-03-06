#include "gpuerrchk.cuh"
#include "assert.h"
#include "real.h"
#include "ch8_in_scan.cuh"
#include <iostream>
#define DAT_SIZE 1024
void test(){
	real X[DAT_SIZE];
	for (int i=0; i< 1024; ++i) X[i]=1;
	real Y[DAT_SIZE];
	real* d_X;
	real* d_Y;
	gpuErrchk(cudaMalloc((void**) &d_X,sizeof(real)*DAT_SIZE));
	gpuErrchk(cudaMemcpy(d_X,X,sizeof(real)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_Y,sizeof(real)*DAT_SIZE ));
	ch8_in_scan(d_X,d_Y,1024);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*DAT_SIZE,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));
	for (int j=0; j<1024; ++j){
		std::cout<< "j=" << j << " Y[j]=" << Y[j] <<std::endl;
		assert(Y[j]==j);
	}
}

int main(){
	test();
}


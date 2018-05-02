#include "gpuerrchk.cuh"
#include "assert.h"
#include "real.h"
#include "ch8_ex_scan.cuh"
#include <iostream>
#define DAT_SIZE 8
void test(){
	real X[DAT_SIZE];
	real Y[DAT_SIZE];
	real* d_X;
	real* d_Y;
	gpuErrchk(cudaMalloc((void**) &d_X,sizeof(real)*DAT_SIZE));
	gpuErrchk(cudaMemcpy(d_X,X,sizeof(real)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_Y,sizeof(real)*DAT_SIZE ));
	ch8_ex_scan(d_X,d_Y,8);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*DAT_SIZE,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));

}

int main(){
	test();
}


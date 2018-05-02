#include "gpuerrchk.cuh"
#include "assert.h"
#include "real.h"
#include "ch8_longer_scan.cuh"
#include <iostream>
void test(){
	constexpr size_t DAT_SIZE=1024*2;
	real X[DAT_SIZE];
	for (int i=0; i< DAT_SIZE; ++i) X[i]=1;
	real Y[DAT_SIZE];
	real* d_X;
	real* d_Y;
	real* d_S;
	gpuErrchk(cudaMalloc((void**) &d_X,sizeof(real)*DAT_SIZE));
	gpuErrchk(cudaMemcpy(d_X,X,sizeof(real)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_Y,sizeof(real)*DAT_SIZE ));	
	gpuErrchk(cudaMalloc((void**) &d_S,sizeof(real)*DAT_SIZE ));
	ch8_longer_scan(d_X,d_Y,d_S,DAT_SIZE, DAT_SIZE/1024, 1024);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*DAT_SIZE,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));
	gpuErrchk(cudaFree(d_S));
	for (int j=0; j<DAT_SIZE; ++j){
		//std::cout<< "j=" << j << " Y[j]=" << Y[j] <<std::endl;
		assert(Y[j]==j+1);
	}
}

void test2(){
	constexpr size_t DAT_SIZE=1024*100;
	real X[DAT_SIZE];
	for (int i=0; i< DAT_SIZE; ++i) X[i]=1;
	real Y[DAT_SIZE];
	real* d_X;
	real* d_Y;
	real* d_S;
	gpuErrchk(cudaMalloc((void**) &d_X,sizeof(real)*DAT_SIZE));
	gpuErrchk(cudaMemcpy(d_X,X,sizeof(real)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_Y,sizeof(real)*DAT_SIZE ));	
	gpuErrchk(cudaMalloc((void**) &d_S,sizeof(real)*DAT_SIZE ));
	ch8_longer_scan(d_X,d_Y,d_S,DAT_SIZE, DAT_SIZE/1024, 1024);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*DAT_SIZE,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));
	gpuErrchk(cudaFree(d_S));
	for (int j=0; j<DAT_SIZE; ++j){
		//std::cout<< "j=" << j << " Y[j]=" << Y[j] <<std::endl;
		assert(Y[j]==j+1);
	}
}

int main(){
	test();
	test2();
	std::cout<< "Success!!!" <<std::endl;
}


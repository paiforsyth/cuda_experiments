#include "gpuerrchk.cuh"
#include "assert.h"
#include "real.h"
#include "ch8_stream_scan.cuh"
#include <iostream>
void test(){
	constexpr size_t DAT_SIZE=2048;
	real X[DAT_SIZE];
	for (int i=0; i< DAT_SIZE; ++i) X[i]=1;
	real Y[DAT_SIZE];
	real* d_X;
	real* d_Y;
	real* d_S;
	int* d_flags;
	int* d_DCounter;
	gpuErrchk(cudaMalloc((void**) &d_X,sizeof(real)*DAT_SIZE));
	gpuErrchk(cudaMemcpy(d_X,X,sizeof(real)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_Y,sizeof(real)*DAT_SIZE ));	
	gpuErrchk(cudaMalloc((void**) &d_S,sizeof(real)*DAT_SIZE ));
	gpuErrchk(cudaMalloc((void**) &d_flags, sizeof(int)*DAT_SIZE ));
	gpuErrchk(cudaMalloc((void**) &d_DCounter,sizeof(int)));
	gpuErrchk( cudaMemset(d_flags,0, sizeof(int)*DAT_SIZE) );
	gpuErrchk( cudaMemset(d_DCounter,0, sizeof(int)) );	
	ch8_stream_scan(d_X,d_Y,d_S,d_flags,d_DCounter,DAT_SIZE, DAT_SIZE/1024, 1024);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*DAT_SIZE,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));
	gpuErrchk(cudaFree(d_S));
	gpuErrchk(cudaFree(d_flags));
	gpuErrchk(cudaFree(d_DCounter));
	
	for (int j=0; j<DAT_SIZE; ++j){
		//std::cout<< "j=" << j << " Y[j]=" << Y[j] <<std::endl;
		assert(Y[j]==j+1);
	}
}


void test2(){
	constexpr size_t DAT_SIZE=2048*100;
	real X[DAT_SIZE];
	for (int i=0; i< DAT_SIZE; ++i) X[i]=1;
	real Y[DAT_SIZE];
	real* d_X;
	real* d_Y;
	real* d_S;
	int* d_flags;
	int* d_DCounter;
	gpuErrchk(cudaMalloc((void**) &d_X,sizeof(real)*DAT_SIZE));
	gpuErrchk(cudaMemcpy(d_X,X,sizeof(real)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_Y,sizeof(real)*DAT_SIZE ));	
	gpuErrchk(cudaMalloc((void**) &d_S,sizeof(real)*DAT_SIZE ));
	gpuErrchk(cudaMalloc((void**) &d_flags, sizeof(int)*DAT_SIZE ));
	gpuErrchk(cudaMalloc((void**) &d_DCounter,sizeof(int)));
	gpuErrchk( cudaMemset(d_flags,0, sizeof(int)*DAT_SIZE) );
	gpuErrchk( cudaMemset(d_DCounter,0, sizeof(int)) );	
	ch8_stream_scan(d_X,d_Y,d_S,d_flags,d_DCounter,DAT_SIZE, DAT_SIZE/1024, 1024);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*DAT_SIZE,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));
	gpuErrchk(cudaFree(d_S));
	gpuErrchk(cudaFree(d_flags));
	gpuErrchk(cudaFree(d_DCounter));
	
	for (int j=0; j<DAT_SIZE; ++j){
		//std::cout<< "j=" << j << " Y[j]=" << Y[j] <<std::endl;
		assert(Y[j]==j+1);
	}
}

void test3(){
	std::cout<<"in test 3 body" <<std::endl<<std::flush;
	constexpr size_t DAT_SIZE=1024*1024;
std::cout<<"Before X allocation" <<std::endl;
	real* X=new real[DAT_SIZE];
	std::cout<<"after X allocation" <<std::endl;
	for (size_t i=0; i< DAT_SIZE; ++i) X[i]=1;
	std::cout<<"after X initialization" <<std::endl;
	real Y[DAT_SIZE];
	real* d_X;
	real* d_Y;
	real* d_S;
	int* d_flags;
	int* d_DCounter;
	gpuErrchk(cudaMalloc((void**) &d_X,sizeof(real)*DAT_SIZE));
	std::cout << "before d_X allocated" << std::endl;
	gpuErrchk(cudaMemcpy(d_X,X,sizeof(real)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_Y,sizeof(real)*DAT_SIZE ));	
	gpuErrchk(cudaMalloc((void**) &d_S,sizeof(real)*DAT_SIZE/1024 ));
	gpuErrchk(cudaMalloc((void**) &d_flags, sizeof(int)*DAT_SIZE/1024 ));
	gpuErrchk(cudaMalloc((void**) &d_DCounter,sizeof(int)));
	gpuErrchk( cudaMemset(d_flags,0, sizeof(int)*DAT_SIZE/1024) );
	gpuErrchk( cudaMemset(d_DCounter,0, sizeof(int)) );	
	std::cout<< "Before scan called" <<std::endl;
	ch8_stream_scan(d_X,d_Y,d_S,d_flags,d_DCounter,DAT_SIZE, DAT_SIZE/1024, 1024);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*DAT_SIZE,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));
	gpuErrchk(cudaFree(d_S));
	gpuErrchk(cudaFree(d_flags));
	gpuErrchk(cudaFree(d_DCounter));
	delete[] X;
	for (int j=0; j<DAT_SIZE; ++j){
		//std::cout<< "j=" << j << " Y[j]=" << Y[j] <<std::endl;
		assert(Y[j]==j+1);
	}
}



int main(){
	test();
	
	std::cout<< "Success in test 1!!!" <<std::endl;
	test2();
	
	std::cout<< "Success in test 2!!!" <<std::endl;
	test3();
	std::cout<< "Success in test 3!!!" <<std::endl;
}


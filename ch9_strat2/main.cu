#include "gpuerrchk.cuh"
#include "assert.h"
#include "real.h"
#include "ch9_strat2.cuh"
#include <iostream>
#include "rstring.h"
#include <string>
#include "cuda_profiler_api.h"
#define DAT_SIZE 1000000000


void test(){
	std::string buffer_string = rstring(DAT_SIZE);
    const char* buffer= buffer_string.c_str();	
	unsigned int histo[]= {0,0,0,0,0,0,0};
	 char* d_buffer;
	unsigned int* d_histo;
	gpuErrchk(cudaMalloc((void**) &d_buffer,sizeof(char)*DAT_SIZE));
	gpuErrchk(cudaMemcpy(d_buffer,buffer,sizeof(char)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_histo,sizeof(int)*7 ));
	
	ch9_strat2(d_buffer,d_histo, (size_t) DAT_SIZE);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(histo,d_histo,sizeof(int)*7,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_buffer));
	gpuErrchk(cudaFree(d_histo));
}

int main(){
	test();
	std::cout<<"DONE!" <<std::endl;
	cudaProfilerStop();
}


#include "gpuerrchk.cuh"
#include "assert.h"
#include "real.h"
#include "ch9_aastrat1.cuh"
#include <iostream>
#define DAT_SIZE 41 
//void ch9_aastrat1(char* buffer, int* histo,size_t inputsize);
void test(){
	unsigned char buffer[]="programming massively parallel processors";
	unsigned int histo[]= {0,0,0,0,0,0,0};
	unsigned char* d_buffer;
	unsigned int* d_histo;
	gpuErrchk(cudaMalloc((void**) &d_buffer,sizeof(char)*DAT_SIZE));
	gpuErrchk(cudaMemcpy(d_buffer,buffer,sizeof(char)*DAT_SIZE,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &d_histo,sizeof(int)*7 ));
	
	ch9_aastrat1(d_buffer,d_histo, (size_t) DAT_SIZE);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(histo,d_histo,sizeof(int)*7,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_buffer));
	gpuErrchk(cudaFree(d_histo));
	for (int i=0; i<7 ;i++){
		std::cout << "histogram bucket " << i << " has value " << histo[i] << std::endl;
	
	}

	

}

int main(){
	test();
}


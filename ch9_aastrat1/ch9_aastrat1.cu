#include "gpuerrchk.cuh"
#include "real.h"

__global__ void ch9_aastrat1_kernel(char* buffer,unsigned int* histo, size_t inputsize){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int section_size = (inputsize-1) / (blockDim.x*gridDim.x) + 1;
	int start=i*section_size;


	for (int k=0; k< section_size; ++k){
		if (start+k < inputsize){
			int alphabet_position=buffer[start+k]-'a';
			if (alphabet_position >=0 && alphabet_position < 26)
				atomicAdd(&histo[alphabet_position/4], 1);
		}
	}
}
void ch9_aastrat1(char* buffer, unsigned int* histo,size_t inputsize){
	ch9_aastrat1_kernel<<<1, 512 >>>(buffer,histo,inputsize);
	gpuErrchk(cudaPeekAtLastError());
}

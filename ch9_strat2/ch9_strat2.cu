#include "gpuerrchk.cuh"
#include "real.h"

__global__ void ch9_strat2_kernel(char* buffer,unsigned int* histo, size_t inputsize){

	unsigned int tid = threadIdx.x +blockIdx.x* blockDim.x;
	for (int i=tid; i< inputsize; i+= blockDim.x*gridDim.x  ){
		int alphabet_position=buffer[i]-'a';
		if (alphabet_position >=0 && alphabet_position < 26)
			atomicAdd(&histo[alphabet_position/4], 1);
	}
}
void ch9_strat2(char* buffer, unsigned int* histo,size_t inputsize){
	ch9_strat2_kernel<<<1, 512 >>>(buffer,histo,inputsize);
	gpuErrchk(cudaPeekAtLastError());
}

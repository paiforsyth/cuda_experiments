#include "gpuerrchk.cuh"
#include "real.h"

__global__ void ch9_strat3_kernel(char* buffer,unsigned int* histo, size_t inputsize,unsigned int num_bins){

	unsigned int tid = threadIdx.x +blockIdx.x* blockDim.x;
	extern __shared__ unsigned int histo_s[];
	for (unsigned int  j=threadIdx.x; j<num_bins; j+=blockDim.x)	
		histo_s[j]=0;		
	__syncthreads();
	for (unsigned int i=tid; i< inputsize; i+= blockDim.x*gridDim.x  ){
		int alphabet_position=buffer[i]-'a';
		if (alphabet_position >=0 && alphabet_position < 26)
			atomicAdd(&histo_s[alphabet_position/4], 1);
	}
	__syncthreads();
	for (unsigned int j=threadIdx.x; j<num_bins; j+=blockDim.x)
		atomicAdd(&histo[j], histo_s[j]);
}
void ch9_strat3(char* buffer, unsigned int* histo,size_t inputsize, unsigned int num_bins){
	//key question: how many thread blocks do we use??
	ch9_strat3_kernel<<<50, 100, num_bins * sizeof(int) >>>(buffer,histo,inputsize, num_bins); 
	gpuErrchk(cudaPeekAtLastError());
}

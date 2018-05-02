#include "real.h"
#include "gpuerrchk.cuh"
#include "reduce.cuh"
//kernel used to sum an array containing size elements, where size can be 2^0,2^1,..,2^10
//intended to be called as a single thread block with size threads

//Note: here is the reason that the below operation never reads outside the bounds of partialsum if it has length 2^k.
//since t%(2*stride)==0 we have that  t=m*2^{i+1} for some m and where we are in iteration i (stride=2^i).  Since t is in the bounds of the array,
//t<2^k. 
//Thus 2m<2^{k-i} (*).
//But then t+2^i<2^{k} if and only if (2m+1)<2^{k-i}.  This holds since both sides of equation * are even.
__global__
void kreducev1(real* X, real* d_ans, unsigned int size){
	extern __shared__ real partialsum[];	
	unsigned int t= threadIdx.x;
	partialsum[t]=X[t];
	for (unsigned int  stride = 1; stride < size; stride *= 2 ){
		__syncthreads();
		if (t % (2*stride) == 0)
			partialsum[t]+=partialsum[t+stride];
	}
	if (t == 0)
		*d_ans=partialsum[0];
}

__global__
void kreducev2(real* X, real* d_ans, unsigned int size){
	extern __shared__ real partialsum[];	
	unsigned int t= threadIdx.x;
	partialsum[t]=X[t];
	for (unsigned int stride = size/2; stride >= 1; stride = stride >> 1){
		__syncthreads();
		if (t < stride)
			partialsum[t] += partialsum[t+stride];
	}
	
	if (t == 0)
		*d_ans=partialsum[0];

}

real reducev1(real* X, unsigned int numel){
	int memsize= sizeof(real)*numel;
	real* d_X;
	real* d_ans;
	gpuErrchk(cudaMalloc( (void**) &d_X, memsize));
	gpuErrchk(cudaMalloc( (void**) &d_ans, sizeof(real)  ) );
	gpuErrchk(cudaMemcpy(d_X, X, memsize, cudaMemcpyHostToDevice));
	kreducev1<<<1, numel, memsize>>>(d_X,d_ans,numel);
	gpuErrchk( cudaPeekAtLastError() );	
	gpuErrchk( cudaFree(d_X));
	real ans;
	gpuErrchk(cudaMemcpy(&ans,d_ans,sizeof(real),cudaMemcpyDeviceToHost ));
	gpuErrchk(cudaFree(d_ans));
	return ans;
}



real reducev2(real* X, unsigned int numel){
	int memsize= sizeof(real)*numel;
	real* d_X;
	real* d_ans;
	gpuErrchk(cudaMalloc( (void**) &d_X, memsize));
	gpuErrchk(cudaMalloc( (void**) &d_ans, sizeof(real)  ) );
	gpuErrchk(cudaMemcpy(d_X, X, memsize, cudaMemcpyHostToDevice));
	kreducev2<<<1, numel, memsize>>>(d_X,d_ans,numel);
	gpuErrchk( cudaPeekAtLastError() );	
	gpuErrchk( cudaFree(d_X));
	real ans;
	gpuErrchk(cudaMemcpy(&ans,d_ans,sizeof(real),cudaMemcpyDeviceToHost ));
	gpuErrchk(cudaFree(d_ans));
	return ans;
}

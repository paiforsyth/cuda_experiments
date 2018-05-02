#include "gpuerrchk.cuh"
#include "real.h"
#define SECTION_SIZE 512
__global__ void ch13_quadtree_kernel(real* X, real* Y, int inputsize){
}

void ch13_quadtree(real* d_X, real* d_Y,int inputsize){
	ch13_quadtree_kernel<<<ceil(inputsize/ (real) SECTION_SIZE),SECTION_SIZE>>>(d_X,d_Y,inputsize);
	gpuErrchk(cudaPeekAtLastError());
}

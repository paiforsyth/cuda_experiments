#!/bin/bash

if [[ -n $1 ]]; then 
#create the exercise directory
mkdir ./$1
#copy the prototype files
cp ./prototype/gpuerrchk.cuh ./$1/
cp ./prototype/real.h ./$1/
#create the build directory
mkdir ./$1/build
#create the kernel file
cat << EOF  > ./$1/$1.cu
#include "gpuerrchk.cuh"
#include "real.h"
#define SECTION_SIZE 512
__global__ void $1_kernel(real* X, real* Y, int inputsize){
}

void $1(real* d_X, real* d_Y,int inputsize){
	$1_kernel<<<ceil(inputsize/ (real) SECTION_SIZE),SECTION_SIZE>>>(d_X,d_Y,inputsize);
	gpuErrchk(cudaPeekAtLastError());
}
EOF

#create the header file
cat << EOF > ./$1/$1.cuh
#include "real.h"
void $1(real* d_X, real* d_Y,int inputsize);
EOF

#create the makefile
cat << EOF > ./$1/Makefile
FLAGS = -std=c++14 -I~/Dropbox/src/

main : main.cu ./build/$1.o $1.cuh
	nvcc \$(FLAGS) -o main main.cu ./build/$1.o


test : test.cu ./build/$1.o $1.cuh
	nvcc \$(FLAGS) -o test test.cu ./build/$1.o


./build/$1.o : $1.cu
	nvcc \$(FLAGS) -c -o ./build/$1.o $1.cu

EOF

#create the MAIN file
cat << EOF > ./$1/main.cu
#include "gpuerrchk.cuh"
#include "assert.h"
#include "real.h"
#include "$1.cuh"
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
	$1(d_X,d_Y,DAT_SIZE);
	gpuErrchk( cudaPeekAtLastError() );		
	gpuErrchk(cudaMemcpy(Y,d_Y,sizeof(real)*DAT_SIZE,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Y));

}

int main(){
	test();
}

EOF

#start the test file as a copy of the main file
cp ./$1/main.cu ./$1/test.cu

else
	echo "Please provide a name for the exercise "
fi

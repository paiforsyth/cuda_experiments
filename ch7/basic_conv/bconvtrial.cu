#include "basic_conv.cuh"
#include "assert.h"
#include "real.h"
#include <iostream>


void trial(){
	constexpr int asize=10^5;
	constexpr int bsize=1000;
	real A[asize];
	for(int i=0; i< asize; i++){
		A[i]=1;
	}
	real M[bsize];
	for (int i=0; i<bsize; ++i){
		M[i]=i;	
	}
	real P[asize];
	basic_conv(A,M,P,bsize,asize);
}

int main(){
	trial();
	
}

#include <assert.h>
#include "shareconvtrial.cuh"
#include "real.h"
#include <iostream>

void trial(){
	constexpr int asize=1000;
	constexpr int bsize=11;
	real A[asize];
	for(int i=0; i< asize; i++){
		A[i]=1;
	}
	real M[bsize];
	for (int i=0; i<bsize; ++i){
		M[i]=i;	
	}
	real P[asize];
	share_conv(A,M,P,bsize,asize);
	//for (int i=0; i<asize; ++i)
	//	std::cout << "P[" << i << "]=" << P[i] <<std::endl;
	//assert(P[20]==55);
}

int main(){
	trial();
	//std::cout << "success!!!!" <<std::endl;
}


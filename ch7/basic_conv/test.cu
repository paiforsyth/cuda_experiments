#include "basic_conv.cuh"
#include "assert.h"
#include "real.h"
#include <iostream>
#include "const_conv.cuh"
void test1(){
	real A[]={1,2,3,4,5,6,7};
	real M[]={3,4,5,4,3};
	real P[7];
	basic_conv(A,M,P,5,7);
	for(int i=0; i<7; i++)
		std::cout<< "P[" <<i << "]="<<P[i] <<std::endl;

	assert(P[1]==38);
	
}
void test2(){
	real A[]={1,2,3,4,5,6,7};
	real M[]={3,4,5,4,3};
	real P[7];
	constant_conv(A,M,P,5,7);
	for(int i=0; i<7; i++)
		std::cout<< "P[" <<i << "]="<<P[i] <<std::endl;

	assert(P[1]==38);
	
}
int main(){
	test1();
	test2();	
	std::cout<< "success!\n" <<std::flush;
}

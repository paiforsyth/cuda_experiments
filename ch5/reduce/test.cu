#include "reduce.cuh"
#include "real.h"
#include "assert.h"
#include <iostream>
void sumTest(){
	real summands[1024];
	for (int i=0; i!=1024; ++i) 
		summands[i]=1;
	assert(reducev1(summands,1024) == 1024);
	assert(reducev2(summands,1024) == 1024);
}

int main(){
	sumTest();
	std::cout << "Success!!!\n" << std::flush; 
}

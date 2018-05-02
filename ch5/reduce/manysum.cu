#include "reduce.cuh"
#include "real.h"
#include "assert.h"
#include <iostream>
int main(){
	real summands[1024];
	for (int i=0; i!=1024; ++i)
		summands[i]=1;
	for (int j=0; j!=1000; ++j)
		reducev2(summands,1024);

}

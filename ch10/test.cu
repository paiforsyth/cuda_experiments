#include "gMat.cuh"
#include "real.h"
#include <assert.h>
#include <iostream>



void basicMultTest(int w){
	std::vector<real> adat={1,2,3,4};
	std::vector<real> bdat={2,0,0,2,0,2};
	std::vector<real> cdat={0,0,0,0,0,0};
	gMat A{adat,2,2};//A= [1 2; 3 4]
	gMat B{bdat,2,3};//B= [2 0 0; 2 0 2]
	gMat C{cdat,2,3};
	
	int mem=2*w*w*sizeof(real);
	
	
	prod(A,B,C,mem);
	std::cout << C << std::endl;
	assert( C.entry(0,0) == 6 );
	assert(C.entry(0,1) == 0 );
	assert(C.entry(0,2) == 4 );
	assert(C.entry(1,0) == 14 );
	assert(C.entry(1,1) == 0 );
	assert(C.entry(1,2) == 8 );
	A.cleanup();
	B.cleanup();
	C.cleanup();

}

void basicMultTestv2(int w, int s){
	std::vector<real> adat={1,2,3,4};
	std::vector<real> bdat={2,0,0,2,0,2};
	std::vector<real> cdat={0,0,0,0,0,0};
	gMat A{adat,2,2};//A= [1 2; 3 4]
	gMat B{bdat,2,3};//B= [2 0 0; 2 0 2]
	gMat C{cdat,2,3};
	
	
	
	prodv2(A, B, C, w, s);
	std::cout << C << std::endl;
	assert( C.entry(0,0) == 6 );
	assert(C.entry(0,1) == 0 );
	assert(C.entry(0,2) == 4 );
	assert(C.entry(1,0) == 14 );
	assert(C.entry(1,1) == 0 );
	assert(C.entry(1,2) == 8 );
	A.cleanup();
	B.cleanup();
	C.cleanup();

}

int main(){
	basicMultTest(1);
/*	basicMultTest(1);
	basicMultTest(2);
	basicMultTestv2(16,1);
	basicMultTestv2(1,2);
	basicMultTestv2(1,3);
	basicMultTestv2(2,1);
	basicMultTestv2(2,2);
*/	
	
	
	std::cout << "SUCCESS!" << std::endl;
}

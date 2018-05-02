#include <iostream>
#include "gMat.cuh"
#include <vector>
#include "mytime.h"
#include <functional>
#include <math.h>
#include <stdlib.h> 
#include <string>
#include "real.h"
void manyMult(gMat& A, gMat& B, gMat& C,int iter,int mem){
	for (int i=0; i!=iter; ++i){
		prod(A,B,C,mem);
		prod(A,C,B,mem);
	}
}

void manyMultv2(gMat& A, gMat& B, gMat& C,int iter,int tile_size, int stretch ){
	for (int i=0; i!=iter; ++i){
		prodv2(A,B,C,tile_size,stretch);
		prodv2(A,C,B,tile_size,stretch);
	}
}

void multstat(int mem, int row, int col){
	std::cout << "multiplying "  << row << " by "<< col <<" matrices.\n";
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	std::cout << "using "<< mem << " of "<<prop.sharedMemPerBlock << "shared memory\n";
	int w=floor(sqrt(mem/2/sizeof(real)));
	std::cout << "Tile sizes will be: "<< w<< " by "<< w << "\n";
	std::cout << "This amounts to: " << w*w << " threads \n";
	std::cout <<  ceil(row/ (real) w)*ceil(col/ (real) w) << " tiles will be used "<<std::endl;
}

int main(int argc, char *argv[]){
	int w;
	if ( argc >= 2) w=atoi(argv[1]);
	else w=20;
	int r=1000;
	int c=1000;
	gMat A=eye(r);
	gMat B=randgMat(r,c);
	gMat C=randgMat(r,c);
//	std::cout<< "(5,5) entry of A is: "<< A.entry(5,5)<< std::endl;
	A.name="A";
	B.name="B";
	C.name="C";
	//int mem=2*w*w*sizeof(real);
	auto mm=std::bind(manyMultv2,A,B,C,300,w,4);
	//multstat(mem,r,c);
	mm();
	A.cleanup();
	B.cleanup();
	C.cleanup();
	 
}




void test1(){
	std::vector<real> adat={1,2,3,4};
	std::vector<real> bdat={2,0,0,2,0,2};
	std::vector<real> cdat={0,0,0,0,0,0};
	gMat A{adat,2,2};
	gMat B{bdat,2,3};
	gMat C{cdat,2,3};
	std::cout <<"A:\n" << A << std::flush;
	std::cout <<"B:\n" << B << std::flush;
	int mem=8*sizeof(real);
	prod(A,B,C,mem);
	std::cout <<"C:\n" << C << std::flush;
}

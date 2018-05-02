#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <memory>
#include <random>
#include "gMat.cuh"
#include "gpuerrchk.cuh"
#include "real.h"
__global__ void matMulKernel(unsigned int num_rows, real* data, unsigned int* col_index, unsigned int row_p, real* x, real* y){
	unsigned int row =blockIdx.x * blockDim.x +threadIdx.x;
	if (row < num_rows){
		real dot=0;
		unsigned int row_start= row_p[row];
		unsigned int row_end= row_p[row+1];
		for (unsigned int elem=row_start; elem<row_end; ++elem){
			dot+=data[elem]*x[col_index[elem]];
		}	
		y[row]+=dot;
	}
}


gMat::gMat(std::vector<real> datavector, int r, int c,int devnumber): rows{r}, cols{c} {
	assert(datavector.size() == rows*cols );
	int size=sizeof(real)*rows*cols;

	cudaDeviceProp dev_prop;	
	cudaGetDeviceProperties(&dev_prop,devnumber);
	assert(dev_prop.totalGlobalMem>size);

	gpuErrchk(cudaMalloc( (void**)&d_data, size));
	gpuErrchk(cudaMemcpy(d_data, datavector.data(), size, cudaMemcpyHostToDevice));
}

void gMat::cleanup(){
	gpuErrchk(cudaFree(d_data));
}



int gMat::getrows(){ return rows;}
int gMat::getcols(){ return cols;}

real gMat::entry(int i, int j){
	assert(i>=0 && i<rows);
	assert(j>=0 && j<cols);
	real val=0;

	cudaMemcpy(&val, &d_data[i*cols+j],sizeof(real),cudaMemcpyDeviceToHost );
	return val;
}


std::vector<real> gMat::tovector(){
	real* h_data=new real[rows*cols];
	int size=rows*cols*sizeof(real);
	cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);
	std::vector<real> v;
	for (int i=0; i<rows; ++i) for (int j=0; j<cols; ++j){
		v.push_back(h_data[i*cols+j]);
	}
	delete[] h_data;
	return v;
}	

gMat randgMat(int r, int c){
	std::vector<real> datavector;
	std::default_random_engine re;
	std::uniform_real_distribution<real> dist{0.0,1.0};
	for (int i=0; i<r*c; i++) datavector.push_back(dist(re));
	return gMat(datavector,r,c);
}

gMat eye(int r){
	std::vector<real> datavector;
	for (int i=0; i!=r; ++i) for(int j=0; j!=r; ++j) 
		if(i==j) 
			datavector.push_back(1.0);
		else
			datavector.push_back(0.0);
	return gMat(datavector,r,r);
}

void prod(const gMat& A, const gMat& B, gMat& P) {
	int w=16;
	int mem=2*w*w*sizeof(real);
	prod(A,B,P,mem);
}

void prod(const gMat& A, const gMat& B, gMat& P, int mem){
	assert(A.cols==B.rows && A.rows==P.rows && B.cols==P.cols);
	int tile_size= floor(sqrt( mem/2/sizeof(real)));
	assert(tile_size>0);	

	matMulKernel<<<dim3(ceil(B.cols/ (real) tile_size), ceil( A.rows/ (real) tile_size)), dim3(tile_size,tile_size), mem>>>(A.d_data, B.d_data, P.d_data,A.rows, A.cols, B.cols, tile_size );
	gpuErrchk( cudaPeekAtLastError() );	
}

void prodv2(const gMat& A, const gMat& B, gMat& P, int tile_size, int stretch){
	assert(A.cols==B.rows && A.rows==P.rows && B.cols==P.cols);
	assert(tile_size>0 && stretch>0);
	int mem= 2*tile_size*tile_size*stretch*sizeof(real);
	matMulKernelv2 <<<dim3(ceil(B.cols/ (real) tile_size), ceil( A.rows/ (real) tile_size)), dim3(tile_size,tile_size), mem>>>(A.d_data, B.d_data, P.d_data, A.rows, A.cols, B.cols, tile_size, stretch);
	gpuErrchk( cudaPeekAtLastError() );	
}

std::ostream& operator<<(std::ostream& os,  gMat& gm)  
{  
	std::vector<real> v=gm.tovector();
	for (int i=0; i<gm.rows; ++i){
		for (int j=0; j<gm.cols; ++j){
			os << v[i*gm.cols+j] << ",";
		}
		os << "\n";
	}
	return os;  

}  



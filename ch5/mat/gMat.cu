#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <memory>
#include <random>
#include "gMat.cuh"
#include "gpuerrchk.cuh"
#include "real.h"
__global__ void matMulKernel(real* A, real* B, real* P, int m, int  n, int s, int tile_size){
	//each thread in the block will be responsible for a different element of these arrays
	extern __shared__ real tilemem[];
	real* Atile=tilemem;
	real* Btile=&tilemem[tile_size*tile_size];

	int b1=blockIdx.y;
	int b2=blockIdx.x;
	int t1=threadIdx.y;
	int t2=threadIdx.x;

	int row=b1*tile_size+t1;// the indices of the element of P this thread is responsible for computing
	int col=b2*tile_size+t2;

	real Pvalue=0;
	//loop over the tiles necessary to compute the element of P of interest
	for (int j=0; j<ceil( n / (real) tile_size);j++){
		//collaborate with other threads to store the current tiles of A and B
		//this thread is responsible for storing  the (t_1,t_2) element of the (b_1,j) tile of A and the (j,b_2) tile of B
		if( (b1*tile_size+t1) < m && (j*tile_size+t2)<n )
			Atile[tile_size*t1 + t2]=A[ n*(b1*tile_size+t1) + j*tile_size  + t2 ]; // We access A[b1*tile_size+t1][j*tile_size+t2] using linearized indices (A has n columns)
		else
			Atile[tile_size*t1 + t2]=0;
		if( (j*tile_size+t1)<n && (b2*tile_size+t2)<s  )
			Btile[tile_size*t1 + t2]=B[ s*(j*tile_size+t1)  + b2*tile_size + t2 ]; //we access the B[j*tile_size+t1][b2*tile_size+t2] element using linearized indices (B has s columns)
		else
			Btile[tile_size*t1 + t2]=0;
		__syncthreads();

		//sum the elements of the current A and B tiles used in the computation of the element of P for which this thread is responsible
		for (int k=0; k<tile_size; k++)
			Pvalue += Atile[tile_size*t1 + k]*Btile[tile_size*k + t2];

		__syncthreads();

	}
	if(row<m && col<s) P[row*s + col]=Pvalue;

}
//stretched version
//stretches the tiles by a factor of stretch in one direction, so that each thread computes stretch entries of P
//thus more memory need to be allocated when invoking this kernel
__global__ void matMulKernelv2(real* A, real* B, real* P, int m, int  n, int s, int tile_size, int stretch){
	//each thread in the block will be responsible for  different elements of these arrays

	extern __shared__ real tilemem[];
	real* Atile=tilemem;// 2d dimensions of Atile should be thought of as tile_size by stretch*tile_size
	real* Btile=&tilemem[stretch*tile_size*tile_size]; 

	int b1=blockIdx.y;
	int b2=blockIdx.x;
	int t1=threadIdx.y;
	int t2=threadIdx.x;

	int row=b1*tile_size+t1;
	int col=b2*tile_size+t2;

	real Pvalue=0;
	//loop over the tiles necessary to compute the element of P of interest
	for (int j=0; j<ceil( n / (real) tile_size / stretch  );j++){
		//collaborate with other threads to store the current tiles of A and B
		for (int q=0; q != stretch; ++q){

			if( (b1*tile_size+t1) < m && (j*stretch*tile_size + q*tile_size + t2) < n )
				Atile[stretch*tile_size*t1 + q*tile_size + t2]=A[ n*(b1*tile_size+t1) + j*stretch*tile_size + q*tile_size   + t2 ]; // We access A[b1*tile_size+t1][j*stretch*tile_size+q*tile_size+t2] using linearized indices (A has n columns)
			else
				Atile[stretch*tile_size*t1 + q*tile_size + t2]=0;
			if( (j*stretch*tile_size + q*tile_size  +t1)<n && (b2*tile_size+t2)<s  )
				Btile[tile_size*t1 + q*tile_size*tile_size + t2]=B[ s*(j*stretch*tile_size+q*tile_size+t1)  + b2*tile_size + t2 ]; //we access the B[j*stretch*tile_size+q*tile_size+t1][b2*tile_size+t2] element using linearized indices (B has s columns)
			else
				Btile[tile_size*t1 + q*tile_size*tile_size + t2]=0;

		}

		__syncthreads();

		//sum the elements of the current A and B tiles used in the computation of the element of P for which this thread is responsible
		for (int k=0; k<stretch*tile_size; k++)
			Pvalue += Atile[stretch*tile_size*t1 + k]*Btile[tile_size*k + t2];

		__syncthreads();

	}
	if(row<m && col<s) P[row*s + col]=Pvalue;

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



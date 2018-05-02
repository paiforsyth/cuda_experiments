#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <memory>
#include <random>
#include "gMat.cuh"
#include "gpuerrchk.cuh"
__global__ void matMulKernel(float* A, float* B, float* P, int m, int  n, int s, int tile_size){
	//each thread in the block will be responsible for a different element of these arrays
	extern __shared__ float tilemem[];
	float* Atile=tilemem;
	float* Btile=&tilemem[tile_size*tile_size];

	int b1=blockIdx.x;
	int b2=blockIdx.y;
	int t1=threadIdx.x;
	int t2=threadIdx.y;

	int row=b1*tile_size+t1;// the indices of the element of P this thread is responsible for computing
	int col=b2*tile_size+t2;

	float Pvalue=0;
	//loop over the tiles necessary to compute the element of P of interest
	for (int j=0; j<ceil( n / (float) tile_size);j++){
		//collaborate with other threads to store the current tiles of A and B
		//this thread is responsible for storing  the (t_1,t_2) element of the (b_1,j) tile of A and the (j,b_2) tile of B
		if( (b1*tile_size+t1) < m && (j*tile_size+t2)<n )
			Atile[tile_size*t1 + t2]=A[ n*(b1*tile_size+t1) + j*tile_size  + t2 ]; // We access A[b1*tile_size+t1][j*tile_size+t2] using linearized indices (A has n columns)
		if( (j*tile_size+t1)<n && (b2*tile_size+t2)<s  )
			Btile[tile_size*t1 + t2]=B[ s*(j*tile_size+t1)  + b2*tile_size + t2 ]; //we access the B[j*tile_size+t1][b2*tile_size+t2] element using linearized indices (B has s columns)
		__syncthreads();

		//sum the elements of the current A and B tiles used in the computation of the element of P for which this thread is responsible
		for (int k=0; k<tile_size; k++)
			Pvalue += Atile[tile_size*t1 + k]*Btile[tile_size*k + t2];

		__syncthreads();

	}
	if(row<m && col<s) P[row*s + col]=Pvalue;

}

gMat::gMat(std::vector<float> datavector, int r, int c,int devnumber): rows{r}, cols{c} {
	assert(datavector.size() == rows*cols );
	int size=sizeof(float)*rows*cols;
	
	cudaDeviceProp dev_prop;	
	cudaGetDeviceProperties(&dev_prop,devnumber);
	assert(dev_prop.totalGlobalMem>size);

	gpuErrchk(cudaMalloc( (void**)&d_data, size));
	gpuErrchk(cudaMemcpy(d_data, datavector.data(), size, cudaMemcpyHostToDevice));
}

void gMat::cleanup()
{
	gpuErrchk(cudaFree(d_data));
}


//gMat::~gMat(){ 
	//	std::cout <<"de-allocating " << name << '\n' <<std::flush;	
	//	gpuErrchk(cudaFree(d_data));
//}
int gMat::getrows(){ return rows;}
int gMat::getcols(){ return cols;}

std::vector<float> gMat::tovector(){
	float* h_data=new float[rows*cols];
	int size=rows*cols*sizeof(float);
	cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);
	std::vector<float> v;
	for (int i=0; i<rows; ++i) for (int j=0; j<cols; ++j){
		v.push_back(h_data[i*cols+j]);
	}
	delete[] h_data;
	return v;
}	

gMat randgMat(int r, int c){
	std::vector<float> datavector;
	std::default_random_engine re;
	std::uniform_real_distribution<float> dist{0.0,1.0};
	for (int i=0; i<r*c; i++) datavector.push_back(dist(re));
	return gMat(datavector,r,c);
}

gMat eye(int r){
	std::vector<float> datavector;
	for (int i=0; i!=r; ++i) for(int j=0; j!=r; ++j) 
		if(i==j) 
			datavector.push_back(1.0);
		else
			datavector.push_back(0.0);
	return gMat(datavector,r,r);
}

void prod(const gMat& A, const gMat& B, gMat& P, int mem){
	assert(A.cols==B.rows && A.rows==P.rows && B.cols==P.cols);
	int tile_size= floor(sqrt( mem/2/sizeof(float)));
    assert(tile_size>0);	
	
	matMulKernel<<<dim3(ceil(A.rows/ (float) tile_size), ceil( B.cols/ (float) tile_size)), dim3(tile_size,tile_size), mem>>>(A.d_data, B.d_data, P.d_data,A.rows, A.cols, B.cols, tile_size );
	gpuErrchk( cudaPeekAtLastError() );	
}

std::ostream& operator<<(std::ostream& os,  gMat& gm)  
{  
	std::vector<float> v=gm.tovector();
	for (int i=0; i<gm.rows; ++i){
		for (int j=0; j<gm.cols; ++j){
			os << v[i*gm.cols+j] << ",";
		}
		os << "\n";
	}
	return os;  

}  



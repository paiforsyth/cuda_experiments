#include <iostream>
const int TILE_SIZE=2;
//assume that A is m by n and B is n by s
//n is the inner dimension  shared by the two matrices
//A  and B will be broken into blocks of TILE_SIZE by TILE_SIZE
//P will be computed in blocks of TILE_SIZE by TILE_SIZE.  These blocks can be indexed by (b1,b2) where  0<=b1<m/TILE_SIZE and 0<=b2<n/TILE_SIZE.
//problem: if the matrix is not square, tile
__global__ void matMulKernel(float* A, float* B, float* P, int  n, int s){
	//each thread in the block will be responsible for a different element of these arrays
	__shared__ float Atile[TILE_SIZE][TILE_SIZE];
	__shared__ float Btile[TILE_SIZE][TILE_SIZE];
	
	int b1=blockIdx.x;
	int b2=blockIdx.y;
	int t1=threadIdx.x;
	int t2=threadIdx.y;

	int row=b1*TILE_SIZE+t1;// the indices of the element of P this thread is responsible for computing
	int col=b2*TILE_SIZE+t2;
	
	float Pvalue=0;
	//loop over the tiles necessary to compute the element of P of interest
	for (int j=0; j< n / TILE_SIZE;j++){
		//collaborate with other threads to store the current tiles of A and B
		//this thread is responsible for storing  the (t_1,t_2) element of the (b_1,j) tile of A and the (j,b_2) tile of B
		Atile[t1][t2]=A[ n*(b1*TILE_SIZE+t1) + j*TILE_SIZE  + t2 ]; // We access A[b1*TILE_SIZE+t1][j*TILE_SIZE+t2] using linearized indices (A has n columns)
		Btile[t1][t2]=B[ s*(j*TILE_SIZE+t1)  + b2*TILE_SIZE + t2 ]; //we access the B[j*TILE_SIZE+t1][b2*TILE_SIZE+t2] element using linearized indices (B has s columns)
		__syncthreads();

		//sum the elements of the current A and B tiles used in the computation of the element of P for which this thread is responsible
		for (int k=0; k<TILE_SIZE; k++){
			Pvalue += Atile[t1][k]*Btile[k][t2];
		}
		__syncthreads();

	}
	P[row*s + col]=Pvalue;

}
	

void printMat(float* A,int rows, int cols){
	for (int i=0; i<rows; ++i){
	for (int j=0; j< cols; ++j){
		std::cout << A[i*cols+j] << ",";
	}
		std::cout << "\n";
	}
	std::cout << std::endl;
}
void matMul(float* A, float* B, float* P, int m, int n, int s){		
	printMat(A,m,n);
	std::cout << "times \n";
	printMat(B,n,s);
	std::cout << " equals \n";
	float* d_A;
	float* d_B;
	float* d_P;
	cudaMalloc ( (void**)&d_A, m*n*sizeof(float) );
	cudaMalloc (  )
}
int main(){
	float A[]={1,2,3,4};
	float B[]={5,6,7,8};
	float P[4];
	

}

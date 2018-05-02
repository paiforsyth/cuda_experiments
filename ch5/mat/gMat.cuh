#include <vector>
#include <memory>
#include <string>
#include "real.h"
__global__ void matMulKernel(real* A, real* B, real* P, int m, int  n, int s, int tile_size);
class gMat {
	private:
		real* d_data;
		int rows;
		int cols;
	public:
		std::string name;
		gMat(std::vector<real> data, int r, int c,int dev=0); 	
		void cleanup();
	 	int getrows();
	    int getcols();
		real entry(int i, int j);
		std::vector<real> tovector();					
		friend std::ostream& operator<<(std::ostream& os,  gMat& gm);  
		friend void prod(const gMat& A, const gMat& B, gMat& C, int mem);
		friend void prod(const gMat& A, const gMat& B, gMat& P); 	
		friend void prodv2(const gMat& A, const gMat& B, gMat& C, int tile_size, int stretch);

};

gMat randgMat(int r, int c);
gMat eye(int r);

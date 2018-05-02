#include <vector>
#include <memory>
#include <string>
#include "real.h"
__global__ void matMulKernel(unsigned int num_rows, real* data, unsigned int col_index, unsigned it row_p, real* x, real* y)
class csrgMat {
	private:
		real* d_data;
		unsigned  int num_rows;
		unsigned int* col_index;
		unsigned int* row_p;
		
	public:
		std::string name;
		csrgMat(std::vector<real> data, int r, int c,int dev=0); 	
		void cleanup();
	 	int getrows();
	    int getcols();
		real entry(int i, int j);
		std::vector<real> tovector();					
		friend std::ostream& operator<<(std::ostream& os,  csrgMat& gm);  
		friend void prod(const csrgMat& A, const gMat& B, gMat& C, int mem);
		friend void prod(const csrgMat& A, const gMat& B, gMat& P); 	
		friend void prodv2(const csrgMat& A, const gMat& B, gMat& C, int tile_size, int stretch);

};

csrgMat randcsrgMat(int r, int c);
csrgMat eye(int r);

#include <vector>
#include <memory>
#include <string>
__global__ void matMulKernel(float* A, float* B, float* P, int m, int  n, int s, int tile_size);
class gMat {
	private:
		float* d_data;
		int rows;
		int cols;
	public:
		std::string name;
		gMat(std::vector<float> data, int r, int c,int dev=0); 	
		void cleanup();
		//~gMat();
	 	int getrows();
	    int getcols();
		std::vector<float> tovector();					
		friend std::ostream& operator<<(std::ostream& os,  gMat& gm);  
		friend void prod(const gMat& A, const gMat& B, gMat& C, int mem);
};

gMat randgMat(int r, int c);
gMat eye(int r);

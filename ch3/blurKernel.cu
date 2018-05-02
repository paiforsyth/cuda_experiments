#include "ppm.h"
const int BLUR_SIZE=10;
//used to blur a 2d color ppm image
__global__	
void blurKernel(int* in, int* out,int w,int h){
	int curpix=blockIdx.x*blockDim.x+threadIdx.x;
	int row=curpix / w;
	int col=curpix % w;
	if ( row>=h ) return;
	int pixr=0;
	int pixg=0;
	int pixb=0;
	int pixels=0;
	for (int br=-BLUR_SIZE; br<=BLUR_SIZE;br++){
	for (int bc=-BLUR_SIZE; bc<=BLUR_SIZE;bc++){
		int currow=row+br;
		int curcol=col+bc;
		if(currow>=0 && currow<h && curcol>=0 && curcol <w ){
			pixels++;
			int pdex= 3*(currow*w+curcol);
			pixr+=in[pdex];
			pixg+=in[pdex+1];
			pixb+=in[pdex+2];
		}
	}
	}
	int dex=3*(row*w+col);
	out[dex]= round((float)pixr/pixels);
	out[dex+1]= round((float)pixg/pixels);
	out[dex+2]= round((float)pixb/pixels);
}

int main(){
	ppm football("football.ppm");
	int numpixels=football.height*football.width;
	int size=3*numpixels;
	int arsize=sizeof(int)*size;
	std::cout <<"Size is: "<< size;
	int* d_football_data;
	int* d_bfootball_data;
	cudaMalloc((void**)&d_football_data,arsize);
	cudaMalloc((void**)&d_bfootball_data,arsize );
	cudaMemcpy(d_football_data,football.data,arsize,cudaMemcpyHostToDevice);
	cudaMemcpy(d_bfootball_data,football.data,arsize,cudaMemcpyHostToDevice);

	blurKernel<<<ceil(numpixels/256) ,256>>>(d_football_data,d_bfootball_data,football.width,football.height);

	ppm bfootball(football);
	cudaMemcpy(bfootball.data,d_bfootball_data,arsize,cudaMemcpyDeviceToHost);
	bfootball.write("bfootball.ppm");
	cudaFree(d_football_data);
	cudaFree(d_bfootball_data);
}

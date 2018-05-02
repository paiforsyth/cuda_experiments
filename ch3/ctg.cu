#include "ppm.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
__global__
void colorToGreyScaleConversion(int* imdata,int* outimdata,int size){
	int dex= 3*(threadIdx.x+blockIdx.x*blockDim.x);

	if (dex>= size) return;
	int r=imdata[dex];
	int g=imdata[dex+1];
	int b=imdata[dex+2];
	int grey= round(255*( 0.21*(r/255.0)+0.71*(g/255.0)+0.07*(b/255.0)));
//	printf("Grey value is : ")
	outimdata[dex]=grey;
	outimdata[dex+1]=grey;
	outimdata[dex+2]=grey;
	
}
int main(){
	ppm football("football.ppm");
	int size=3*football.height*football.width;
	int arsize=sizeof(int)*size;
	std::cout <<"Size is: "<< size;
	int* d_football_data;
	int* d_gfootball_data;
	cudaMalloc((void**)&d_football_data,arsize);
	cudaMalloc((void**)&d_gfootball_data,arsize );
	cudaMemcpy(d_football_data,football.data,arsize,cudaMemcpyHostToDevice);
	cudaMemcpy(d_gfootball_data,football.data,arsize,cudaMemcpyHostToDevice);

	colorToGreyScaleConversion<<< 1<<20 ,256>>>(d_football_data,d_gfootball_data,size);
	ppm gfootball(football);
	cudaMemcpy(gfootball.data,d_gfootball_data,arsize,cudaMemcpyDeviceToHost);
	gfootball.write("gfootball.ppm");
	cudaFree(d_gfootball_data);
	cudaFree(d_football_data);
}

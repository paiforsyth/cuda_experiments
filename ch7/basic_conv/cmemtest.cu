#include<iostream>
__constant__ float M[10];

int main(){
	float h_M[]={1,2,3,4,5,7,8,9,0};
    cudaMemcpyToSymbol(M,h_M,10*sizeof(float));	
	std::cout<< "yo"<<std::endl;
}

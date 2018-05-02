#include <iostream>
int main(){
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	cudaDeviceProp dev_prop;
	for (int i=0; i<dev_count; i++){
		cudaGetDeviceProperties(&dev_prop,i);
		std::cout << "Device number: " << i << "\n"; 
		std::cout << "Shared memory per block:" << dev_prop.sharedMemPerBlock  << "bytes \n";
		std::cout << "Global Memory:" << dev_prop.totalGlobalMem << "bytes";
	}
	std::cout<< std::flush;
}

FLAGS = -std=c++14





test : test.cu ./build/shareconvtrial.o real.h 
	nvcc $(FLAGS)  -o test ./build/shareconvtrial.o test.cu

./build/shareconvtrial.o : shareconvtrial.cu real.h
	nvcc $(FLAGS) -c -o ./build/shareconvtrial.o shareconvtrial.cu
	


cconvtrial : cconvtrial.cu 
	nvcc $(FLAGS) -o cconvtrial cconvtrial.cu

./build/const_conv.o : const_conv.cu real.h const_conv.cuh
	nvcc $(FLAGS) -c -o ./build/const_conv.o const_conv.cu

FLAGS = -std=c++14

bconvtrial : bconvtrial.cu ./build/basic_conv.o basic_conv.cuh real.h
	nvcc $(FLAGS) -o bconvtrial bconvtrial.cu ./build/basic_conv.o




test : test.cu ./build/basic_conv.o ./build/const_conv.o basic_conv.cuh real.h 
	nvcc $(FLAGS)  -o test ./build/basic_conv.o ./build/const_conv.o test.cu

./build/basic_conv.o : basic_conv.cu real.h
	nvcc $(FLAGS) -c -o ./build/basic_conv.o basic_conv.cu

./build/const_conv.o : const_conv.cu real.h const_conv.cuh
	nvcc $(FLAGS) -c -o ./build/const_conv.o const_conv.cu

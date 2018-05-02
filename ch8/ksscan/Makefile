FLAGS = -std=c++14

test : test.cu  ./build/ksscan.o ksscan.cuh
	nvcc $(FLAGS)  -o test test.cu ./build/ksscan.o

./build/ksscan.o : ksscan.cu
	nvcc $(FLAGS) -c -o ./build/ksscan.o ksscan.cu


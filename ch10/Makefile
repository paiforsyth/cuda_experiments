FLAGS = -std=c++14

gMatTest : gMatTest.cu  ./build/gMat.o gMat.cuh  real.h
	nvcc $(FLAGS)  -o gMatTest gMatTest.cu ./build/gMat.o

./build/gMat.o : gMat.cu gMat.cuh real.h
	nvcc $(FLAGS) -c -o ./build/gMat.o gMat.cu

test : test.cu ./build/gMat.o gMat.cuh real.h 
	nvcc $(FLAGS)  -o test ./build/gMat.o test.cu

FLAGS = -std=c++14

gMatTest : gMatTest.cu  ./build/gMat.o mytime.h
	nvcc $(FLAGS)  -o gMatTest gMatTest.cu ./build/gMat.o

./build/gMat.o : gMat.cu gMat.cuh
	nvcc $(FLAGS) -c -o ./build/gMat.o gMat.cu

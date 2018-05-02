FLAGS = -std=c++14
manysum : manysum.cu ./build/reduce.o reduce.cuh real.h
	nvcc $(FLAGS) -o manysum ./build/reduce.o manysum.cu

test : test.cu ./build/reduce.o  reduce.cuh real.h 
	nvcc ${FLAGS} -o test ./build/reduce.o test.cu

./build/reduce.o : reduce.cu reduce.cuh real.h
	nvcc ${FLAGS} -c -o ./build/reduce.o reduce.cu

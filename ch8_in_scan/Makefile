FLAGS = -std=c++14

main : main.cu ./build/ch8_in_scan.o ch8_ex_scan.cuh
	nvcc $(FLAGS) -o main main.cu ./build/ch8_in_scan.o


test : test.cu ./build/ch8_in_scan.o ch8_ex_scan.cuh
	nvcc $(FLAGS) -o test test.cu ./build/ch8_in_scan.o


./build/ch8_in_scan.o : ch8_ex_scan.cu
	nvcc $(FLAGS) -c -o ./build/ch8_in_scan.o ch8_ex_scan.cu


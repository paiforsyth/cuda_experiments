FLAGS = -std=c++14 -I~/Dropbox/src/

main : main.cu ./build/ch9_aastrat1.o ch9_aastrat1.cuh ./build/rstring.o
	nvcc $(FLAGS) -o main main.cu ./build/ch9_aastrat1.o ./build/rstring.o


test : test.cu ./build/ch9_aastrat1.o ch9_aastrat1.cuh
	nvcc $(FLAGS) -o test test.cu ./build/ch9_aastrat1.o


./build/ch9_aastrat1.o : ch9_aastrat1.cu
	nvcc $(FLAGS) -c -o ./build/ch9_aastrat1.o ch9_aastrat1.cu
./build/rstring.o : rstring.cpp
	nvcc $(FLAGS) -c -o ./build/rstring.o rstring.cpp 

FLAGS = -std=c++14 -I~/Dropbox/src/

main : main.cu ./build/ch9_strat3.o ch9_strat3.cuh ./build/rstring.o
	nvcc $(FLAGS) -o main main.cu ./build/ch9_strat3.o ./build/rstring.o


test : test.cu ./build/ch9_strat3.o ch9_strat3.cuh ./build/rstring.o
	nvcc $(FLAGS) -o test test.cu ./build/ch9_strat3.o ./build/rstring.o 


./build/ch9_strat3.o : ch9_strat3.cu
	nvcc $(FLAGS) -c -o ./build/ch9_strat3.o ch9_strat3.cu
./build/rstring.o : rstring.cpp
	nvcc $(FLAGS) -c -o ./build/rstring.o rstring.cpp 

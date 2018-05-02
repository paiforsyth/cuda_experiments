FLAGS = -std=c++14 -I/home/peter/Dropbox/src/  -L/home/peter/Dropbox/src/  #-I~/Dropbox/src

main : main.cu ./build/ch8_longer_scan.o ch8_longer_scan.cuh 
	nvcc $(FLAGS) -o main main.cu ./build/ch8_longer_scan.o /home/peter/Dropbox/src/next_pow_2.o


test : test.cu ./build/ch8_longer_scan.o ch8_longer_scan.cuh 
	nvcc $(FLAGS) -o test test.cu ./build/ch8_longer_scan.o /home/peter/Dropbox/src/next_pow_2.o


./build/ch8_longer_scan.o : ch8_longer_scan.cu  
	nvcc $(FLAGS) -c -o ./build/ch8_longer_scan.o ch8_longer_scan.cu


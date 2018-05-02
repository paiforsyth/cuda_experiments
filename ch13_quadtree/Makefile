FLAGS = -std=c++14 -I~/Dropbox/src/

main : main.cu ./build/ch13_quadtree.o ch13_quadtree.cuh
	nvcc $(FLAGS) -o main main.cu ./build/ch13_quadtree.o


test : test.cu ./build/ch13_quadtree.o ch13_quadtree.cuh
	nvcc $(FLAGS) -o test test.cu ./build/ch13_quadtree.o

./build/ch13_quadtree.o : ch13_quadtree.cu
	nvcc $(FLAGS) -c -o ./build/ch13_quadtree.o ch13_quadtree.cu

points_test : points_test.cu ./build/points.o points.cuh
	nvcc $(FLAGS) -o points_test points_test.cu ./build/points.o

./build/points.o : points.cu
	nvcc $(FLAGS) -c -o ./build/points.o points.cu

bounding_box_test : bounding_box_test.cu ./build/bounding_box.o bounding_box.cuh
	nvcc $(FLAGS) -o bounding_box_test bounding_box_test.cu ./build/bounding_box.o

./build/bounding_box.o : bounding_box.cu
	nvcc $(FLAGS) -c -o ./build/bounding_box.o bounding_box.cu 

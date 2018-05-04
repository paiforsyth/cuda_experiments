FLAGS = -std=c++14 -I~/Dropbox/src/






ch13_quadtree : ch13_quadtree.cu
	nvcc $(FLAGS)  -o ch13_quadtree ch13_quadtree.cu

points_test : points_test.cu ./build/points.o points.cuh
	nvcc $(FLAGS) -o points_test points_test.cu ./build/points.o

./build/points.o : points.cu
	nvcc $(FLAGS) -c -o ./build/points.o points.cu

bounding_box_test : bounding_box_test.cu ./build/bounding_box.o bounding_box.cuh
	nvcc $(FLAGS) -o bounding_box_test bounding_box_test.cu ./build/bounding_box.o

./build/bounding_box.o : bounding_box.cu
	nvcc $(FLAGS) -c -o ./build/bounding_box.o bounding_box.cu 

./build/quad_tree_node.o : quad_tree_node.cu quad_tree_node.cuh boundind_box.cuh
	nvcc $(FLAGS) -c -o ./build/quad_tree_node.o quad_tree_node.cu

./build/parameters.o : parameters.cu parameters.cuh
	nvcc $(FLAGS) -c -o ./build/parameters.o parameters.cu
